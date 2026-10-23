pub mod commands;
pub mod events;
pub mod message;

use lavalink_rs::{model::events as lava_events, prelude::*};
use poise::serenity_prelude as serenity;
use songbird::SerenityInit;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;
use tracing::{error, info, warn};

pub struct Data {
    pub lavalink: Arc<OnceCell<LavalinkClient>>,
    pub prefix: String,
}

impl Data {
    pub fn lavalink(&self) -> Option<LavalinkClient> {
        self.lavalink.get().cloned()
    }
}

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Context<'a> = poise::Context<'a, Data, Error>;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,lavalink_rs=info")),
        )
        .init();

    // Preload the MiniDiscordGPT model on startup.
    let _ = tokio::task::spawn_blocking(|| {
        let _ = message::get_brain();
    })
    .await;

    let guild = serenity::GuildId::new(851831355229601844);

    let discord_token =
        std::env::var("DISCORD_TOKEN").expect("Missing DISCORD_TOKEN environment variable");
    let lavalink_host =
        std::env::var("LAVALINK_HOST").unwrap_or_else(|_| "localhost:2333".to_string());
    let lavalink_password =
        std::env::var("LAVALINK_PASSWORD").unwrap_or_else(|_| "youshallnotpass".to_string());
    let lavalink_ssl = std::env::var("LAVALINK_SSL")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);
    let prefix = std::env::var("BOT_PREFIX").unwrap_or_else(|_| "!".to_string());
    let setup_prefix = prefix.clone();

    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![
                commands::play(),
                commands::play_local(),
                commands::skip(),
                commands::queue(),
                commands::stop(),
                commands::volume(),
                commands::pause(),
            ],
            event_handler: |ctx, event, _framework_ctx, data| {
                Box::pin(async move {
                    message::message_event(ctx, event, &data.prefix).await;
                    Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
                })
            },
            prefix_options: poise::PrefixFrameworkOptions {
                prefix: Some(prefix),
                ..Default::default()
            },
            ..Default::default()
        })
        .setup(move |ctx, ready, framework| {
            Box::pin(async move {
                info!("Logged in as {}", ready.user.name);

                // Register slash commands in the configured guild (and any other guilds the bot is in)
                // without aborting setup if a guild is inaccessible.
                let mut target_guilds = vec![guild];
                for g in &ready.guilds {
                    if !target_guilds.contains(&g.id) {
                        target_guilds.push(g.id);
                    }
                }
                for g_id in target_guilds {
                    if let Err(err) =
                        poise::builtins::register_in_guild(ctx, &framework.options().commands, g_id)
                            .await
                    {
                        warn!("Could not register commands in guild {}: {}", g_id, err);
                    }
                }

                let bot_user_id: UserId = ctx.cache.current_user().id.into();
                let lavalink_cell = Arc::new(OnceCell::new());
                let lavalink_cell_bg = lavalink_cell.clone();

                // Connect to Lavalink in the background with a timeout so an unresponsive
                // Lavalink node NEVER blocks Poise setup, commands, or message events.
                tokio::spawn(async move {
                    loop {
                        let events = lava_events::Events {
                            raw: Some(events::raw_event),
                            ready: Some(events::ready_event),
                            track_start: Some(events::track_start),
                            track_end: Some(events::track_end),
                            ..Default::default()
                        };

                        let node_local = NodeBuilder {
                            hostname: lavalink_host.clone(),
                            is_ssl: lavalink_ssl,
                            events: lava_events::Events::default(),
                            password: lavalink_password.clone(),
                            user_id: bot_user_id,
                            session_id: None,
                        };

                        info!("Connecting to Lavalink node at {}...", lavalink_host);
                        match tokio::time::timeout(
                            Duration::from_secs(10),
                            LavalinkClient::new(
                                events,
                                vec![node_local],
                                NodeDistributionStrategy::round_robin(),
                            ),
                        )
                        .await
                        {
                            Ok(client) => {
                                let _ = lavalink_cell_bg.set(client);
                                info!("Lavalink client initialized.");
                                break;
                            }
                            Err(_) => {
                                error!(
                                    "Timed out connecting to Lavalink at {}. Retrying in 15s...",
                                    lavalink_host
                                );
                                tokio::time::sleep(Duration::from_secs(15)).await;
                            }
                        }
                    }
                });

                info!("Bot setup complete! Ready to handle commands and messages.");
                Ok(Data {
                    lavalink: lavalink_cell,
                    prefix: setup_prefix,
                })
            })
        })
        .build();

    let intents = serenity::GatewayIntents::non_privileged()
        | serenity::GatewayIntents::GUILD_VOICE_STATES
        | serenity::GatewayIntents::MESSAGE_CONTENT;

    let mut client = serenity::ClientBuilder::new(discord_token, intents)
        .register_songbird()
        .framework(framework)
        .await?;

    client.start().await?;

    Ok(())
}
