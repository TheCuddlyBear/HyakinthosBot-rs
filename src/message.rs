use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder,
};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use regex::Regex;
use serenity::all as serenity;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

// Must match the Python model you exported!
// (If you exported the 256-dim model earlier, change D_MODEL to 256 and MAX_SEQ_LEN to 128)
const MAX_SEQ_LEN: usize = 96;
const D_MODEL: usize = 192;
const N_LAYERS: usize = 4;
const N_HEADS: usize = 4;

// ============================================================================
// 1. TRANSFORMER MODEL ARCHITECTURE
// ============================================================================

struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        let head_dim = d_model / n_heads;
        let qkv = linear_no_bias(d_model, 3 * d_model, vb.pp("qkv"))?;
        let proj = linear_no_bias(d_model, d_model, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            n_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;

        let q = qkv
            .narrow(2, 0, c)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, c, c)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, 2 * c, c)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (1, 1, t, t), x.device())?;
        att = att.broadcast_add(&mask)?;

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;

        Ok(self.proj.forward(&out)?)
    }
}

struct TransformerBlock {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
}

impl TransformerBlock {
    fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        let ln1 = layer_norm(d_model, 1e-5, vb.pp("ln1"))?;
        let attn = CausalSelfAttention::load(vb.pp("attn"), d_model, n_heads)?;
        let ln2 = layer_norm(d_model, 1e-5, vb.pp("ln2"))?;
        let mlp_fc1 = linear(d_model, 4 * d_model, vb.pp("mlp.0"))?;
        let mlp_fc2 = linear(4 * d_model, d_model, vb.pp("mlp.2"))?;
        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp_fc1,
            mlp_fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.attn.forward(&self.ln1.forward(x)?)?;
        let x = (residual + x)?;

        let residual = &x;
        let h = self.mlp_fc1.forward(&self.ln2.forward(&x)?)?;
        let h = h.gelu_erf()?;
        let h = self.mlp_fc2.forward(&h)?;
        Ok((residual + h)?)
    }
}

pub struct MiniDiscordGPT {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    pub max_seq_len: usize,
}

impl MiniDiscordGPT {
    pub fn load(vb: VarBuilder, vocab_size: usize) -> Result<Self> {
        let tok_emb = embedding(vocab_size, D_MODEL, vb.pp("tok_emb"))?;
        let pos_emb = embedding(MAX_SEQ_LEN, D_MODEL, vb.pp("pos_emb"))?;
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            blocks.push(TransformerBlock::load(
                vb.pp(format!("blocks.{i}")),
                D_MODEL,
                N_HEADS,
            )?);
        }
        let ln_f = layer_norm(D_MODEL, 1e-5, vb.pp("ln_f"))?;
        Ok(Self {
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            max_seq_len: MAX_SEQ_LEN,
        })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let pos = Tensor::arange(0u32, t as u32, idx.device())?;

        let tok_embeddings = self.tok_emb.forward(idx)?;
        let pos_embeddings = self.pos_emb.forward(&pos)?;
        let mut x = tok_embeddings.broadcast_add(&pos_embeddings)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.ln_f.forward(&x)?;

        let head_weight = self.tok_emb.embeddings().t()?;
        let logits = x.broadcast_matmul(&head_weight)?;
        Ok(logits)
    }
}

// ============================================================================
// 2. SAFE GLOBAL STATE & INFERENCE ENGINE (NEVER PANICS OR DEADLOCKS)
// ============================================================================

pub struct BotBrain {
    pub model: MiniDiscordGPT,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub re_url: Regex,
    pub re_emoji: Regex,
    pub re_mention: Regex,
    pub re_ws: Regex,
}

impl BotBrain {
    pub fn clean_text(&self, text: &str) -> String {
        let s = self.re_url.replace_all(text, "");
        let s = self.re_emoji.replace_all(&s, "$1");
        let s = self.re_mention.replace_all(&s, "@user");
        self.re_ws.replace_all(&s, " ").trim().to_string()
    }
}

// Stores Result<BotBrain, String> so a missing file NEVER panics or locks the bot!
static BRAIN: OnceLock<std::result::Result<BotBrain, String>> = OnceLock::new();

pub fn get_brain() -> std::result::Result<&'static BotBrain, &'static str> {
    BRAIN
        .get_or_init(|| {
            let cwd = std::env::current_dir().unwrap_or_default();

            let device = Device::Cpu;
            let tokenizer = Tokenizer::from_file("discord_tokenizer.json").map_err(|e| {
                format!(
                    "Could not open 'discord_tokenizer.json' in {}: {e}",
                    cwd.display()
                )
            })?;
            let vocab_size = tokenizer.get_vocab_size(true);

            // Read bytes safely into memory instead of unsafe mmap
            let safetensor_bytes = std::fs::read("discord_mini_gpt.safetensors").map_err(|e| {
                format!(
                    "Could not open 'discord_mini_gpt.safetensors' in {}: {e}",
                    cwd.display()
                )
            })?;

            let vb =
                VarBuilder::from_buffered_safetensors(safetensor_bytes, DType::F32, &device)
                    .map_err(|e| format!("Failed to parse safetensors file: {e}"))?;

            let model = MiniDiscordGPT::load(vb, vocab_size).map_err(|e| {
                format!(
                    "Model weight shape mismatch (check D_MODEL={D_MODEL}, MAX_SEQ_LEN={MAX_SEQ_LEN}, vocab_size={vocab_size}): {e}"
                )
            })?;

            Ok(BotBrain {
                model,
                tokenizer,
                device,
                re_url: Regex::new(r"https?://\S+").unwrap(),
                re_emoji: Regex::new(r"<a?(:[a-zA-Z0-9_]+:)\d+>").unwrap(),
                re_mention: Regex::new(r"<[@#!&]\d+>").unwrap(),
                re_ws: Regex::new(r"\s+").unwrap(),
            })
        })
        .as_ref()
        .map_err(|s| s.as_str())
}

pub fn generate_take(
    brain: &BotBrain,
    user_message: &str,
    max_new_tokens: usize,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    repetition_penalty: f32,
    num_candidates: usize,
) -> Result<String> {
    let tokenizer = &brain.tokenizer;
    let model = &brain.model;
    let device = &brain.device;

    let pad_id = tokenizer
        .token_to_id("<|pad|>")
        .ok_or_else(|| E::msg("Missing <|pad|> token"))? as usize;
    let bos_id = tokenizer
        .token_to_id("<|bos|>")
        .ok_or_else(|| E::msg("Missing <|bos|> token"))?;
    let sep_id = tokenizer
        .token_to_id("<|sep|>")
        .ok_or_else(|| E::msg("Missing <|sep|> token"))?;
    let eos_id = tokenizer
        .token_to_id("<|eos|>")
        .ok_or_else(|| E::msg("Missing <|eos|> token"))?;
    let unk_id = tokenizer
        .token_to_id("<|unk|>")
        .ok_or_else(|| E::msg("Missing <|unk|> token"))? as usize;

    let clean_msg = brain.clean_text(user_message);
    if clean_msg.is_empty() {
        return Ok(String::new());
    }

    let encoding = tokenizer.encode(clean_msg.clone(), false).map_err(E::msg)?;

    let mut prompt_ids = vec![bos_id];
    prompt_ids.extend(encoding.get_ids());
    prompt_ids.push(sep_id);

    let max_prompt = model.max_seq_len.saturating_sub(max_new_tokens).max(16);
    if prompt_ids.len() > max_prompt {
        prompt_ids.truncate(max_prompt - 1);
        prompt_ids.push(sep_id);
    }

    let prompt_len = prompt_ids.len();
    let mut sequences: Vec<Vec<u32>> = vec![prompt_ids; num_candidates];
    let mut finished = vec![false; num_candidates];
    let mut log_probs_sum = vec![0.0f32; num_candidates];
    let mut lengths = vec![0usize; num_candidates];

    let mut rng = rand::rng();

    for step in 0..max_new_tokens {
        let cur_len = sequences[0].len();
        let start_idx = cur_len.saturating_sub(model.max_seq_len);
        let window_len = cur_len - start_idx;

        let flat_input: Vec<u32> = sequences
            .iter()
            .flat_map(|seq| &seq[start_idx..])
            .copied()
            .collect();
        let idx_cond = Tensor::from_vec(flat_input, (num_candidates, window_len), device)?;

        let logits = model.forward(&idx_cond)?;
        let last_logits = logits.i((.., window_len - 1, ..))?;
        let logits_vec: Vec<Vec<f32>> = last_logits.to_vec2()?;

        for b in 0..num_candidates {
            if finished[b] {
                sequences[b].push(eos_id);
                continue;
            }

            let raw_row = &logits_vec[b];

            // Compute true (unwarped) model log-probabilities for accurate quality scoring
            let raw_max = raw_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let raw_log_sum_exp = raw_row
                .iter()
                .map(|&l| (l - raw_max).exp())
                .sum::<f32>()
                .ln()
                + raw_max;

            let mut row = raw_row.clone();

            // Never allow special framing/padding tokens mid-generation
            for &forbidden_id in &[pad_id, unk_id, bos_id as usize, sep_id as usize] {
                if forbidden_id < row.len() {
                    row[forbidden_id] = f32::NEG_INFINITY;
                }
            }
            // Prevent empty or 1-token premature termination
            if step < 2 && (eos_id as usize) < row.len() {
                row[eos_id as usize] = f32::NEG_INFINITY;
            }

            // Count token occurrences in generated slice for repetition + frequency penalty
            let mut token_counts: HashMap<u32, usize> = HashMap::new();
            for &tid in &sequences[b][prompt_len..] {
                *token_counts.entry(tid).or_insert(0) += 1;
            }
            for (token_id, count) in token_counts {
                let tid = token_id as usize;
                if tid < row.len() && row[tid].is_finite() {
                    let penalty = repetition_penalty * (1.0 + 0.15 * (count.saturating_sub(1) as f32));
                    if row[tid] < 0.0 {
                        row[tid] *= penalty;
                    } else {
                        row[tid] /= penalty;
                    }
                    // Additive frequency penalty to discourage loops
                    row[tid] -= 0.25 * (count as f32);
                }
            }

            // Slight temperature spread across the 2 candidates:
            // Candidate 1 (~0.70) is more focused/coherent, Candidate 2 (~0.80) is slightly more expressive
            let candidate_temp = if num_candidates > 1 {
                let factor = 0.92 + 0.14 * (b as f32 / (num_candidates - 1) as f32);
                (temperature * factor).max(1e-5)
            } else {
                temperature.max(1e-5)
            };

            for val in &mut row {
                *val /= candidate_temp;
            }

            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = row.iter().map(|&l| (l - max_logit).exp()).collect();
            let sum_probs: f32 = probs.iter().sum();
            if sum_probs > 0.0 {
                for p in &mut probs {
                    *p /= sum_probs;
                }
            }

            // 1. Min-P filtering: drop tokens unlikely relative to the top token
            let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
            let threshold = min_p * max_prob;
            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }

            // 2. Top-P (nucleus) filtering: keep only the most probable nucleus of tokens
            if top_p > 0.0 && top_p < 1.0 {
                let mut indices: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
                indices.sort_unstable_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut cumulative = 0.0f32;
                for (rank, &idx) in indices.iter().enumerate() {
                    if rank > 0 && cumulative >= top_p {
                        probs[idx] = 0.0;
                    } else {
                        cumulative += probs[idx];
                    }
                }
            }

            let renorm_sum: f32 = probs.iter().sum();
            if renorm_sum > 0.0 {
                for p in &mut probs {
                    *p /= renorm_sum;
                }
            }

            let dist = WeightedIndex::new(&probs)?;
            let next_id = dist.sample(&mut rng) as u32;

            if next_id == eos_id {
                finished[b] = true;
            } else {
                let true_log_prob = raw_row[next_id as usize] - raw_log_sum_exp;
                log_probs_sum[b] += true_log_prob;
                lengths[b] += 1;
            }
            sequences[b].push(next_id);
        }

        if finished.iter().all(|&f| f) {
            break;
        }
    }

    let prompt_lower = clean_msg.to_lowercase();
    let mut candidates: Vec<(f32, String)> = Vec::new();

    for b in 0..num_candidates {
        let gen_slice = &sequences[b][prompt_len..];
        let end_pos = gen_slice
            .iter()
            .position(|&id| id == eos_id)
            .unwrap_or(gen_slice.len());
        let clean_ids = &gen_slice[..end_pos];

        let text = tokenizer
            .decode(clean_ids, true)
            .map_err(E::msg)?
            .trim()
            .to_string();

        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        // Base score: length-normalized true model log-likelihood
        let len_norm = (lengths[b].max(1) as f32).powf(0.75);
        let mut score = log_probs_sum[b] / len_norm;

        // Reward natural completion with <|eos|> vs getting cut off at max_new_tokens
        if finished[b] {
            score += 0.35;
        } else {
            score -= 0.50;
        }

        // Length quality shaping: penalize overly short fragments, reward natural 3..=16 word takes
        if word_count == 0 {
            score -= 100.0;
        } else if word_count < 2 {
            score -= 2.5;
        } else if word_count < 3 {
            score -= 0.8;
        } else if (3..=16).contains(&word_count) {
            score += 0.25;
        }

        // Penalize word-level repetition (e.g. repeated words in the same reply)
        if word_count >= 2 {
            let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
            let diversity_ratio = unique_words.len() as f32 / word_count as f32;
            if diversity_ratio < 0.75 {
                score -= (0.75 - diversity_ratio) * 4.0;
            }
            for pair in words.windows(2) {
                if pair[0].eq_ignore_ascii_case(pair[1]) {
                    score -= 1.5;
                }
            }
        }

        // Penalize verbatim echoing of the user's message
        if !prompt_lower.is_empty() && text.to_lowercase() == prompt_lower {
            score -= 3.0;
        }

        candidates.push((score, text));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(candidates[0].1.clone())
}

// ============================================================================
// 3. NON-BLOCKING SERENITY MESSAGE EVENT HANDLER
// ============================================================================

pub async fn message_event(ctx: &serenity::Context, event: &serenity::FullEvent, bot_prefix: &str) {
    if let serenity::FullEvent::Message { new_message } = event {
        let content = new_message.content.trim();

        // 1. Ignore bots, empty messages, and command prefixes so commands never conflict
        if new_message.author.bot
            || content.is_empty()
            || (!bot_prefix.is_empty() && content.starts_with(bot_prefix))
            || content.starts_with('!')
            || content.starts_with('/')
            || content.starts_with('?')
        {
            return;
        }

        // 2. Currently set to 100% for testing (change `100` back to `3` or set `REPLY_CHANCE=3` in `.env` once working!)
        let chance_percent = std::env::var("REPLY_CHANCE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(100)
            .min(100);

        if chance_percent == 0 || rand::random_range(1..=100) > chance_percent {
            return;
        }

        // 3. Clone handles and spawn in the background so Poise/Serenity commands NEVER get blocked!
        let http = ctx.http.clone();
        let msg = new_message.clone();
        let user_content = content.to_string();
        let author_name = new_message.author.display_name().to_string();

        tokio::spawn(async move {
            let _typing = msg.channel_id.start_typing(&http);

            let generated_result = tokio::task::spawn_blocking(move || {
                let brain = get_brain().map_err(E::msg)?;
                generate_take(
                    brain,
                    &user_content,
                    40,   // max_new_tokens
                    0.75, // base temperature (Candidate 1 @ ~0.69, Candidate 2 @ ~0.80)
                    0.08, // min_p (cuts off low-confidence tail noise)
                    0.90, // top_p (nucleus sampling)
                    1.20, // repetition_penalty (+ frequency scaling)
                    2,    // num_candidates (generates 2 in parallel and picks highest-scoring)
                )
            })
                .await;

            if let Ok(Ok(mut response)) = generated_result {
                if !response.is_empty() {
                    response = response.replace("@user", &author_name);
                    let _ = msg.reply(&http, response).await;
                }
            }
        });
    }
}