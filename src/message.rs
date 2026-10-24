use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use regex::Regex;
use serenity::all as serenity;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;
use tracing::{info, warn};

const DEFAULT_HF_REPO: &str = "TheCuddlyBear/HellGPT";
const LOCAL_MODEL_DIR: &str = "models/HellGPT";

// Default architecture hyperparameters (overridden by config.json from Hugging Face)
const DEFAULT_VOCAB_SIZE: usize = 4096;
const DEFAULT_MAX_SEQ_LEN: usize = 96;
const DEFAULT_D_MODEL: usize = 512;
const DEFAULT_N_LAYERS: usize = 6;
const DEFAULT_N_HEADS: usize = 8;

const NL_WORDS: &[&str] = &[
    "de", "het", "een", "ik", "je", "jij", "wij", "we", "ze", "hij", "zij", "niet", "wel",
    "maar", "want", "omdat", "ook", "nog", "al", "naar", "voor", "met", "van", "op", "aan",
    "uit", "over", "bij", "als", "dan", "toch", "echt", "gewoon", "heel", "veel", "meer",
    "geen", "niks", "iets", "iemand", "niemand", "alles", "hoe", "wat", "waar", "waarom",
    "wanneer", "wie", "welke", "deze", "dit", "die", "dat", "mijn", "jouw", "zijn", "haar",
    "onze", "jullie", "hun", "ben", "bent", "heb", "hebt", "heeft", "hebben", "had", "was",
    "waren", "wordt", "worden", "kan", "kunt", "kunnen", "moet", "moeten", "mag", "wil",
    "willen", "ga", "gaat", "gaan", "kom", "komt", "komen", "doe", "doet", "doen", "zie",
    "ziet", "zien", "zeg", "zegt", "denken", "weet", "weten", "vind", "vinden", "lekker",
    "leuk", "mooi", "goed", "slecht", "dik", "lelijk", "gekke", "facking", "ff", "wrm",
    "gwn", "sws", "eigenlijk", "misschien", "straks", "vanavond", "morgen", "gisteren",
];

const EN_WORDS: &[&str] = &[
    "the", "and", "you", "that", "was", "for", "are", "with", "his", "they", "this",
    "have", "from", "one", "had", "word", "but", "not", "what", "all", "were", "when",
    "your", "can", "said", "there", "use", "each", "which", "she", "how", "their", "will",
    "other", "about", "out", "many", "then", "them", "these", "some", "her", "would",
    "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "see",
    "number", "way", "could", "people", "than", "first", "water", "been", "call", "who",
    "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may",
    "part", "actually", "really", "think", "know", "good", "bad", "yeah", "nah", "bro",
    "dude", "literally", "always", "never", "someone", "anyone", "tonight", "game",
];

// ============================================================================
// 1. TRANSFORMER MODEL ARCHITECTURE (RMSNorm + RoPE + QK-Norm + SwiGLU)
// ============================================================================

struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    fn load(dim: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps)?)
    }
}

fn precompute_rope_cache(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut cos_vec = Vec::with_capacity(seq_len * half_dim);
    let mut sin_vec = Vec::with_capacity(seq_len * half_dim);

    let thetas: Vec<f32> = (0..half_dim)
        .map(|i| 1.0f32 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect();

    for pos in 0..seq_len {
        let p = pos as f32;
        for &theta in &thetas {
            let freq = p * theta;
            cos_vec.push(freq.cos());
            sin_vec.push(freq.sin());
        }
    }

    let cos = Tensor::from_vec(cos_vec, (seq_len, half_dim), device)?;
    let sin = Tensor::from_vec(sin_vec, (seq_len, half_dim), device)?;
    Ok((cos, sin))
}

/// Applies interleaved Rotary Position Embeddings matching `apply_rope` in `model.py`:
/// `x1, x2 = x[..., ::2], x[..., 1::2]`
/// `out = stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)`
fn apply_rope(
    x: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    start_pos: usize,
) -> Result<Tensor> {
    let (b, n_heads, t, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    let cos = rope_cos
        .narrow(0, start_pos, t)?
        .reshape((1, 1, t, half_dim))?;
    let sin = rope_sin
        .narrow(0, start_pos, t)?
        .reshape((1, 1, t, half_dim))?;

    let x_pairs = x.reshape((b, n_heads, t, half_dim, 2))?;
    let x1 = x_pairs.narrow(4, 0, 1)?.squeeze(4)?.contiguous()?;
    let x2 = x_pairs.narrow(4, 1, 1)?.squeeze(4)?.contiguous()?;

    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    Ok(Tensor::stack(&[&out1, &out2], 4)?.reshape((b, n_heads, t, head_dim))?)
}

struct CausalSelfAttention {
    qkv: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        let head_dim = d_model / n_heads;
        let qkv = linear_no_bias(d_model, 3 * d_model, vb.pp("qkv"))?;
        let q_norm = RmsNorm::load(head_dim, 1e-6, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::load(head_dim, 1e-6, vb.pp("k_norm"))?;
        let proj = linear_no_bias(d_model, d_model, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            q_norm,
            k_norm,
            proj,
            n_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        start_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;

        let q = qkv
            .narrow(2, 0, c)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = qkv
            .narrow(2, c, c)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = qkv
            .narrow(2, 2 * c, c)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = apply_rope(&q, rope_cos, rope_sin, start_pos)?;
        let k = apply_rope(&k, rope_cos, rope_sin, start_pos)?;

        let (k, v) = match kv_cache.take() {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[&prev_k, &k], 2)?;
                let v = Tensor::cat(&[&prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if t > 1 {
            let total_t = k.dim(2)?;
            let offset = total_t - t;
            let mask: Vec<f32> = (0..t)
                .flat_map(|i| {
                    (0..total_t).map(move |j| {
                        if j > offset + i {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (1, 1, t, total_t), x.device())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;

        Ok(self.proj.forward(&out)?)
    }
}

struct SwiGLUMLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGLUMLP {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let hidden_dim = ((8 * d_model / 3) + 63) / 64 * 64;
        let w1 = linear_no_bias(d_model, hidden_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(hidden_dim, d_model, vb.pp("w2"))?;
        let w3 = linear_no_bias(d_model, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        Ok(self.w2.forward(&(gate * up)?)?)
    }
}

struct TransformerBlock {
    ln1: RmsNorm,
    attn: CausalSelfAttention,
    ln2: RmsNorm,
    mlp: SwiGLUMLP,
}

impl TransformerBlock {
    fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        let ln1 = RmsNorm::load(d_model, 1e-6, vb.pp("ln1"))?;
        let attn = CausalSelfAttention::load(vb.pp("attn"), d_model, n_heads)?;
        let ln2 = RmsNorm::load(d_model, 1e-6, vb.pp("ln2"))?;
        let mlp = SwiGLUMLP::load(vb.pp("mlp"), d_model)?;
        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        start_pos: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self
            .attn
            .forward(&self.ln1.forward(x)?, rope_cos, rope_sin, start_pos, kv_cache)?;
        let x = (residual + h)?;

        let residual = &x;
        let h = self.mlp.forward(&self.ln2.forward(&x)?)?;
        Ok((residual + h)?)
    }
}

pub struct MiniDiscordGPT {
    tok_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: RmsNorm,
    head_weight: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    pub max_seq_len: usize,
    pub n_layers: usize,
}

impl MiniDiscordGPT {
    pub fn load(
        vb: VarBuilder,
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        device: &Device,
    ) -> Result<Self> {
        // Support both Hugging Face Transformers wrapper ("model.*") and flat safetensors
        let vb = if vb.contains_tensor("model.tok_emb.weight") {
            vb.pp("model")
        } else {
            vb
        };

        let tok_emb = embedding(vocab_size, d_model, vb.pp("tok_emb"))?;
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(TransformerBlock::load(
                vb.pp(format!("blocks.{i}")),
                d_model,
                n_heads,
            )?);
        }
        let ln_f = RmsNorm::load(d_model, 1e-6, vb.pp("ln_f"))?;
        let head_weight = tok_emb.embeddings().t()?.contiguous()?;

        let rope_len = max_seq_len.max(256);
        let (rope_cos, rope_sin) =
            precompute_rope_cache(rope_len, d_model / n_heads, 10000.0, device)?;

        Ok(Self {
            tok_emb,
            blocks,
            ln_f,
            head_weight,
            rope_cos,
            rope_sin,
            max_seq_len,
            n_layers,
        })
    }

    /// Runs a forward pass starting at position `start_pos`, updating `kv_cache`,
    /// and returns logits for ONLY the last token of shape `(b, vocab_size)`.
    pub fn forward_last_logits(
        &self,
        idx: &Tensor,
        start_pos: usize,
        kv_cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let mut x = self.tok_emb.forward(idx)?;

        for (block, cache) in self.blocks.iter().zip(kv_cache.iter_mut()) {
            x = block.forward(&x, &self.rope_cos, &self.rope_sin, start_pos, cache)?;
        }

        let x_last = x.narrow(1, t - 1, 1)?;
        let x_last = self.ln_f.forward(&x_last)?;
        let logits = x_last.broadcast_matmul(&self.head_weight)?.squeeze(1)?;
        Ok(logits)
    }
}

// ============================================================================
// 2. HUGGING FACE DOWNLOADER, SAFE GLOBAL STATE & INFERENCE ENGINE
// ============================================================================

pub struct BotBrain {
    pub model: MiniDiscordGPT,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub re_url: Regex,
    pub re_emoji: Regex,
    pub re_mention: Regex,
    pub re_ws: Regex,
    pub re_words: Regex,
    pub nl_words: HashSet<&'static str>,
    pub en_words: HashSet<&'static str>,
}

impl BotBrain {
    pub fn clean_text(&self, text: &str) -> String {
        let s = self.re_url.replace_all(text, "");
        let s = self.re_emoji.replace_all(&s, "$1");
        let s = self.re_mention.replace_all(&s, "@user");
        self.re_ws.replace_all(&s, " ").trim().to_string()
    }

    /// Detects '<|nl|>' or '<|en|>' from text. Defaults to '<|nl|>' on ties, matching `data.py`.
    pub fn detect_lang(&self, text: &str) -> &'static str {
        let lower = text.to_lowercase();
        let mut nl_score = 0usize;
        let mut en_score = 0usize;
        for m in self.re_words.find_iter(&lower) {
            let w = m.as_str();
            if self.nl_words.contains(w) {
                nl_score += 1;
            }
            if self.en_words.contains(w) {
                en_score += 1;
            }
        }
        if en_score > nl_score {
            "<|en|>"
        } else {
            "<|nl|>"
        }
    }
}

fn resolve_hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
    {
        let trimmed = token.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    let token_path = Path::new(&home).join(".cache").join("huggingface").join("token");
    let token = std::fs::read_to_string(token_path).ok()?;
    let trimmed = token.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Ensures `config.json`, `tokenizer.json`, and `model.safetensors` from `TheCuddlyBear/HellGPT`
/// are available locally, downloading them from Hugging Face if missing.
pub async fn ensure_hf_model_files() -> std::result::Result<PathBuf, String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let required = ["config.json", "tokenizer.json", "model.safetensors"];

    // 1. Check current working directory first
    if required.iter().all(|f| cwd.join(f).is_file()) {
        return Ok(cwd);
    }

    // 2. Check models/HellGPT directory
    let model_dir = cwd.join(LOCAL_MODEL_DIR);
    if required.iter().all(|f| model_dir.join(f).is_file()) {
        return Ok(model_dir);
    }

    std::fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create model cache directory {}: {e}", model_dir.display()))?;

    let repo_id = std::env::var("HF_MODEL_REPO").unwrap_or_else(|_| DEFAULT_HF_REPO.to_string());
    let token = resolve_hf_token();
    let client = reqwest::Client::new();

    for filename in required {
        let dest = model_dir.join(filename);
        if dest.is_file() {
            continue;
        }

        let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
        info!("Downloading {filename} from Hugging Face ({repo_id})...");

        let mut req = client.get(&url);
        if let Some(ref t) = token {
            req = req.bearer_auth(t);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| format!("Failed to download {url}: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Hugging Face returned HTTP {} for {url} (if the repo is private, set HF_TOKEN in .env)",
                resp.status()
            ));
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("Failed reading response body for {filename}: {e}"))?;

        std::fs::write(&dest, &bytes)
            .map_err(|e| format!("Failed writing {} to disk: {e}", dest.display()))?;
    }

    info!("Hugging Face model files ready in {}", model_dir.display());
    Ok(model_dir)
}

static BRAIN: OnceLock<std::result::Result<BotBrain, String>> = OnceLock::new();

fn load_brain_from_dir(model_dir: &Path) -> std::result::Result<BotBrain, String> {
    let device = Device::Cpu;

    let config_path = model_dir.join("config.json");
    let config_json: serde_json::Value = if config_path.is_file() {
        let s = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {e}", config_path.display()))?;
        serde_json::from_str(&s)
            .map_err(|e| format!("Failed to parse {}: {e}", config_path.display()))?
    } else {
        serde_json::json!({})
    };

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        format!(
            "Could not open '{}': {e}",
            tokenizer_path.display()
        )
    })?;

    let vocab_size = config_json
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| tokenizer.get_vocab_size(true).max(DEFAULT_VOCAB_SIZE));
    let max_seq_len = config_json
        .get("max_seq_len")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_MAX_SEQ_LEN);
    let d_model = config_json
        .get("d_model")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_D_MODEL);
    let n_layers = config_json
        .get("n_layers")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_N_LAYERS);
    let n_heads = config_json
        .get("n_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_N_HEADS);

    let safetensors_path = model_dir.join("model.safetensors");
    let safetensor_bytes = std::fs::read(&safetensors_path).map_err(|e| {
        format!(
            "Could not open '{}': {e}",
            safetensors_path.display()
        )
    })?;

    let vb = VarBuilder::from_buffered_safetensors(safetensor_bytes, DType::F32, &device)
        .map_err(|e| format!("Failed to parse safetensors file: {e}"))?;

    let model = MiniDiscordGPT::load(
        vb,
        vocab_size,
        max_seq_len,
        d_model,
        n_layers,
        n_heads,
        &device,
    )
    .map_err(|e| {
        format!(
            "Model weight shape mismatch (vocab_size={vocab_size}, max_seq_len={max_seq_len}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}): {e}"
        )
    })?;

    info!(
        "Loaded HellGPT model (vocab_size={vocab_size}, max_seq_len={max_seq_len}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads})"
    );

    Ok(BotBrain {
        model,
        tokenizer,
        device,
        re_url: Regex::new(r"https?://\S+").unwrap(),
        re_emoji: Regex::new(r"<a?(:[a-zA-Z0-9_]+:)\d+>").unwrap(),
        re_mention: Regex::new(r"<[@#!&]\d+>").unwrap(),
        re_ws: Regex::new(r"\s+").unwrap(),
        re_words: Regex::new(r"[a-zA-Z']+").unwrap(),
        nl_words: NL_WORDS.iter().copied().collect(),
        en_words: EN_WORDS.iter().copied().collect(),
    })
}

/// Async initializer called at startup to download model files (if needed) and preload weights.
pub async fn init_brain() {
    match ensure_hf_model_files().await {
        Ok(model_dir) => {
            let _ = tokio::task::spawn_blocking(move || {
                let res = BRAIN.get_or_init(|| load_brain_from_dir(&model_dir));
                if let Err(e) = res {
                    warn!("Failed to initialize HellGPT brain: {e}");
                }
            })
            .await;
        }
        Err(e) => {
            warn!("Could not prepare Hugging Face model files: {e}");
        }
    }
}

pub fn get_brain() -> std::result::Result<&'static BotBrain, &'static str> {
    BRAIN
        .get_or_init(|| {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let model_dir = if cwd.join("model.safetensors").is_file() {
                cwd
            } else {
                cwd.join(LOCAL_MODEL_DIR)
            };
            load_brain_from_dir(&model_dir)
        })
        .as_ref()
        .map_err(|s| s.as_str())
}

pub fn generate_take(
    brain: &BotBrain,
    raw_turns: &[String],
    max_new_tokens: usize,
    temperature: f32,
    min_p: f32,
    repetition_penalty: f32,
    num_candidates: usize,
) -> Result<String> {
    let tokenizer = &brain.tokenizer;
    let model = &brain.model;
    let device = &brain.device;

    let turns: Vec<String> = raw_turns
        .iter()
        .map(|t| brain.clean_text(t))
        .filter(|t| !t.is_empty())
        .collect();

    if turns.is_empty() {
        return Ok(String::new());
    }

    let last_turn = turns.last().unwrap();
    let lang_tag = brain.detect_lang(last_turn);

    let pad_id = tokenizer
        .token_to_id("<|pad|>")
        .ok_or_else(|| E::msg("Missing <|pad|> token"))?;
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
        .ok_or_else(|| E::msg("Missing <|unk|> token"))?;
    let nl_id = tokenizer
        .token_to_id("<|nl|>")
        .ok_or_else(|| E::msg("Missing <|nl|> token"))?;
    let en_id = tokenizer
        .token_to_id("<|en|>")
        .ok_or_else(|| E::msg("Missing <|en|> token"))?;

    let lang_id = if lang_tag == "<|nl|>" { nl_id } else { en_id };

    // Encode multi-turn history separated by <|sep|>
    let mut trigger_ids: Vec<u32> = Vec::new();
    for (i, turn) in turns.iter().enumerate() {
        if i > 0 {
            trigger_ids.push(sep_id);
        }
        let enc = tokenizer.encode(turn.as_str(), false).map_err(E::msg)?;
        trigger_ids.extend_from_slice(enc.get_ids());
    }

    // Left-truncate older context so we always have room for max_new_tokens
    let max_trig_len = model
        .max_seq_len
        .saturating_sub(max_new_tokens + 4)
        .max(8);
    if trigger_ids.len() > max_trig_len {
        trigger_ids = trigger_ids[trigger_ids.len() - max_trig_len..].to_vec();
    }

    let mut prompt_ids = Vec::with_capacity(trigger_ids.len() + 3);
    prompt_ids.push(bos_id);
    prompt_ids.extend_from_slice(&trigger_ids);
    prompt_ids.push(sep_id);
    prompt_ids.push(lang_id);

    let last_turn_enc = tokenizer
        .encode(last_turn.as_str(), false)
        .map_err(E::msg)?;
    let last_turn_ids: HashSet<u32> = last_turn_enc.get_ids().iter().copied().collect();

    let prompt_len = prompt_ids.len();
    let mut sequences: Vec<Vec<u32>> = vec![prompt_ids.clone(); num_candidates];
    let mut finished = vec![false; num_candidates];
    let mut log_probs_sum = vec![0.0f32; num_candidates];
    let mut lengths = vec![0usize; num_candidates];
    let mut kv_cache: Vec<Option<(Tensor, Tensor)>> = vec![None; model.n_layers];

    let mut rng = rand::rng();
    let temp = temperature.max(1e-5);
    let echo_penalty = 1.0 + (repetition_penalty - 1.0) * 0.35;
    let forbidden_ids = [
        pad_id as usize,
        bos_id as usize,
        sep_id as usize,
        nl_id as usize,
        en_id as usize,
        unk_id as usize,
    ];

    for step in 0..max_new_tokens {
        let cur_len = sequences[0].len();
        if cur_len > model.max_seq_len {
            break;
        }

        let logits_vec: Vec<Vec<f32>> = if step == 0 {
            // Prefill the shared prompt ONCE (batch_size = 1), then replicate the KV cache
            let idx_cond = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), device)?;
            let last_logits = model.forward_last_logits(&idx_cond, 0, &mut kv_cache)?;
            let single_row: Vec<f32> = last_logits.i(0)?.to_vec1()?;

            if num_candidates > 1 {
                for layer_cache in &mut kv_cache {
                    if let Some((k, v)) = layer_cache.take() {
                        let k_rep = k.repeat((num_candidates, 1, 1, 1))?;
                        let v_rep = v.repeat((num_candidates, 1, 1, 1))?;
                        *layer_cache = Some((k_rep, v_rep));
                    }
                }
            }
            vec![single_row; num_candidates]
        } else {
            let last_tokens: Vec<u32> =
                sequences.iter().map(|seq| *seq.last().unwrap()).collect();
            let idx_cond = Tensor::from_vec(last_tokens, (num_candidates, 1), device)?;
            let last_logits =
                model.forward_last_logits(&idx_cond, cur_len - 1, &mut kv_cache)?;
            last_logits.to_vec2()?
        };

        for b in 0..num_candidates {
            if finished[b] {
                sequences[b].push(eos_id);
                continue;
            }

            let mut row: Vec<f32> = logits_vec[b].iter().map(|&v| v / temp).collect();

            // Penalize repeating tokens already generated (and lightly penalize echoing the prompt)
            let generated_so_far: HashSet<u32> =
                sequences[b][prompt_len..].iter().copied().collect();

            for &token_id in &generated_so_far {
                let tid = token_id as usize;
                if tid < row.len() {
                    if row[tid] < 0.0 {
                        row[tid] *= repetition_penalty;
                    } else {
                        row[tid] /= repetition_penalty;
                    }
                }
            }

            for &token_id in last_turn_ids.difference(&generated_so_far) {
                let tid = token_id as usize;
                if tid < row.len() {
                    if row[tid] < 0.0 {
                        row[tid] *= echo_penalty;
                    } else {
                        row[tid] /= echo_penalty;
                    }
                }
            }

            for &special_id in &forbidden_ids {
                if special_id < row.len() {
                    row[special_id] = f32::NEG_INFINITY;
                }
            }

            for val in &mut row {
                if val.is_nan() {
                    *val = -1e4;
                } else if *val == f32::INFINITY {
                    *val = 1e4;
                }
            }

            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = row
                .iter()
                .map(|&l| {
                    if l == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (l - max_logit).exp()
                    }
                })
                .collect();
            let sum_probs: f32 = probs.iter().sum();
            if sum_probs > 0.0 {
                for p in &mut probs {
                    *p /= sum_probs;
                }
            }

            // Min-P filtering matching `generate.py`
            let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
            let threshold = min_p * max_prob;
            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }

            let mut row_sum: f32 = probs.iter().sum();
            if row_sum <= 1e-8 {
                probs.fill(0.0);
                let top_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                probs[top_idx] = 1.0;
                row_sum = 1.0;
            }

            for p in &mut probs {
                *p /= row_sum;
            }

            let dist = WeightedIndex::new(&probs)?;
            let next_id = dist.sample(&mut rng) as u32;
            let token_log_prob = (probs[next_id as usize] + 1e-10).ln();

            log_probs_sum[b] += token_log_prob;
            lengths[b] += 1;
            sequences[b].push(next_id);

            if next_id == eos_id {
                finished[b] = true;
            }
        }

        if finished.iter().all(|&f| f) {
            break;
        }
    }

    let last_turn_lower = last_turn.to_lowercase();
    let mut candidates: Vec<(f32, String)> = Vec::with_capacity(num_candidates);

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

        let word_count = text.split_whitespace().count();
        let len_norm = (lengths[b].max(1) as f32).powf(0.8);
        let mut score = log_probs_sum[b] / len_norm;

        if word_count < 3 {
            score -= 10.0;
        }
        if brain.detect_lang(&text) != lang_tag {
            score -= 3.5;
        }
        if text.to_lowercase() == last_turn_lower {
            score -= 15.0;
        }

        candidates.push((score, text));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(candidates[0].1.clone())
}

// ============================================================================
// 3. MULTI-TURN CONTEXT EXTRACTION & SERENITY MESSAGE EVENT HANDLER
// ============================================================================

const MERGE_WINDOW_SEC: i64 = 60;
const REPLY_WINDOW_SEC: i64 = 180;
const MAX_CONTEXT_TURNS: usize = 3; // 2 previous messages + 1 message to reply to

#[derive(Clone)]
struct CleanedTurn {
    ids: Vec<serenity::MessageId>,
    author_id: serenity::UserId,
    content: String,
    timestamp: i64,
    ref_id: Option<serenity::MessageId>,
}

fn is_valid_discord_message(
    msg: &serenity::Message,
    clean_content: &str,
    bot_prefix: &str,
) -> bool {
    if msg.author.bot {
        return false;
    }
    if !matches!(
        msg.kind,
        serenity::MessageType::Regular | serenity::MessageType::InlineReply
    ) {
        return false;
    }
    let raw = msg.content.trim();
    if (!bot_prefix.is_empty() && raw.starts_with(bot_prefix))
        || raw.starts_with('!')
        || raw.starts_with('/')
        || raw.starts_with('$')
        || raw.starts_with('.')
        || raw.starts_with('-')
        || raw.starts_with('?')
        || raw.starts_with("p!")
        || raw.starts_with("m!")
        || raw.starts_with(">>")
    {
        return false;
    }
    clean_content.len() >= 3
}

/// Builds up to 3 turns (`[prev_2, prev_1, target_message]`) from recent channel history,
/// mirroring `DiscordDataProcessor` in `data.py`.
/// Context turns do NOT need to be from the same author.
async fn collect_context_turns(
    http: &serenity::Http,
    target_msg: &serenity::Message,
    bot_prefix: &str,
) -> Vec<String> {
    let mut raw_messages: Vec<serenity::Message> = match target_msg
        .channel_id
        .messages(
            http,
            serenity::GetMessages::new()
                .before(target_msg.id)
                .limit(15),
        )
        .await
    {
        Ok(mut msgs) => {
            msgs.reverse(); // Chronological order (oldest -> newest)
            msgs
        }
        Err(_) => Vec::new(),
    };
    raw_messages.push(target_msg.clone());

    let Ok(brain) = get_brain() else {
        return vec![target_msg.content.trim().to_string()];
    };

    let mut cleaned_msgs: Vec<CleanedTurn> = Vec::new();
    for m in &raw_messages {
        let clean = brain.clean_text(&m.content);
        if m.id != target_msg.id && !is_valid_discord_message(m, &clean, bot_prefix) {
            continue;
        }
        if clean.is_empty() {
            continue;
        }
        let ref_id = m.message_reference.as_ref().and_then(|r| r.message_id);
        cleaned_msgs.push(CleanedTurn {
            ids: vec![m.id],
            author_id: m.author.id,
            content: clean,
            timestamp: m.timestamp.unix_timestamp(),
            ref_id,
        });
    }

    if cleaned_msgs.is_empty() {
        return Vec::new();
    }

    // Merge consecutive messages by the same author within 60s when not an explicit reply,
    // matching `DiscordDataProcessor` in `data.py`.
    let mut merged: Vec<CleanedTurn> = Vec::new();
    for m in cleaned_msgs {
        let should_merge = if let Some(last) = merged.last() {
            last.author_id == m.author_id
                && (m.timestamp - last.timestamp) <= MERGE_WINDOW_SEC
                && m.ref_id.is_none()
        } else {
            false
        };

        if should_merge {
            let last = merged.last_mut().unwrap();
            last.content.push(' ');
            last.content.push_str(&m.content);
            last.timestamp = m.timestamp;
            last.ids.extend(m.ids);
        } else {
            merged.push(m);
        }
    }

    let last_idx = merged.len() - 1;
    let target_turn = &merged[last_idx];

    // Map every message ID to its index in `merged` for explicit reply chain lookup
    let mut id_to_idx: HashMap<serenity::MessageId, usize> = HashMap::new();
    for (idx, turn) in merged.iter().enumerate() {
        for &id in &turn.ids {
            id_to_idx.insert(id, idx);
        }
    }

    // 1. If the message being replied to is an explicit Discord Reply, follow the reply chain first
    if target_turn.ref_id.is_some() {
        let mut chain_indices = vec![last_idx];
        let mut curr_idx = last_idx;
        while let Some(ref_id) = merged[curr_idx].ref_id {
            if chain_indices.len() >= MAX_CONTEXT_TURNS {
                break;
            }
            if let Some(&parent_idx) = id_to_idx.get(&ref_id) {
                if parent_idx < curr_idx && !chain_indices.contains(&parent_idx) {
                    chain_indices.push(parent_idx);
                    curr_idx = parent_idx;
                    continue;
                }
            }
            break;
        }

        // If the explicit reply chain only had 1 parent (2 turns total), also include the
        // preceding turn before that parent if available so we still get 2 context messages
        if chain_indices.len() < MAX_CONTEXT_TURNS {
            let oldest_idx = *chain_indices.last().unwrap();
            let mut back_idx = oldest_idx;
            while chain_indices.len() < MAX_CONTEXT_TURNS && back_idx > 0 {
                let prev_idx = back_idx - 1;
                if (merged[back_idx].timestamp - merged[prev_idx].timestamp) <= REPLY_WINDOW_SEC {
                    chain_indices.push(prev_idx);
                    back_idx = prev_idx;
                } else {
                    break;
                }
            }
        }

        chain_indices.reverse();
        return chain_indices
            .into_iter()
            .map(|i| merged[i].content.clone())
            .collect();
    }

    // 2. Sliding window over recent channel turns (up to 2 previous turns + target turn)
    let mut window_indices = vec![last_idx];
    let mut curr_idx = last_idx;
    while window_indices.len() < MAX_CONTEXT_TURNS && curr_idx > 0 {
        let prev_idx = curr_idx - 1;
        if (merged[curr_idx].timestamp - merged[prev_idx].timestamp) <= REPLY_WINDOW_SEC {
            window_indices.push(prev_idx);
            curr_idx = prev_idx;
        } else {
            break;
        }
    }

    // Fallback: if no messages were within the 180s window, still take up to 2 immediately
    // preceding messages from the channel if present so the bot has context when possible
    if window_indices.len() == 1 && last_idx > 0 {
        let start = last_idx.saturating_sub(MAX_CONTEXT_TURNS - 1);
        return merged[start..=last_idx]
            .iter()
            .map(|t| t.content.clone())
            .collect();
    }

    window_indices.reverse();
    window_indices
        .into_iter()
        .map(|i| merged[i].content.clone())
        .collect()
}

static GEN_SEMAPHORE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);

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

        // 2. Reply probability check (default 3%, configurable via REPLY_CHANCE in .env)
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
        let prefix_owned = bot_prefix.to_string();
        let author_name = new_message.author.display_name().to_string();

        tokio::spawn(async move {
            let Ok(_permit) = GEN_SEMAPHORE.acquire().await else {
                return;
            };
            let _typing = msg.channel_id.start_typing(&http);

            let turns = collect_context_turns(&http, &msg, &prefix_owned).await;
            if turns.is_empty() {
                return;
            }

            let generated_result = tokio::task::spawn_blocking(move || {
                let brain = get_brain().map_err(E::msg)?;
                generate_take(
                    brain,
                    &turns,
                    40,   // max_new_tokens (matches generate.py)
                    0.85, // temperature (matches generate.py)
                    0.06, // min_p (matches generate.py)
                    1.25, // repetition_penalty (matches generate.py)
                    5,    // num_candidates (matches generate.py)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hellgpt_matches_python_and_generates() {
        let brain = get_brain().expect("Failed to load HellGPT brain");
        let prompt: Vec<u32> = vec![
            1, 348, 331, 476, 209, 883, 203, 2, 2355, 376, 271, 1135, 360, 2, 473, 211, 257, 408,
            298, 511, 2, 5,
        ];
        let mut kv_cache = vec![None; brain.model.n_layers];
        let idx = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &brain.device).unwrap();
        let logits = brain
            .model
            .forward_last_logits(&idx, 0, &mut kv_cache)
            .unwrap();
        let row: Vec<f32> = logits.i(0).unwrap().to_vec1().unwrap();

        // Compare against PyTorch reference values:
        // id=83: 11.656191, id=442: 9.806259, id=51: 9.509355
        assert!((row[83] - 11.656191).abs() < 1e-3, "got {}", row[83]);
        assert!((row[442] - 9.806259).abs() < 1e-3, "got {}", row[442]);
        assert!((row[51] - 9.509355).abs() < 1e-3, "got {}", row[51]);

        let turns = vec![
            "ik heb echt honger".to_string(),
            "wat ga je eten dan".to_string(),
            "weet ik nog niet man".to_string(),
        ];
        let reply = generate_take(brain, &turns, 40, 0.85, 0.06, 1.25, 5).unwrap();
        println!("Generated multi-turn reply: {reply}");
        assert!(!reply.is_empty());
    }
}
