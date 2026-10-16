pub mod commands;
pub mod events;

use lavalink_rs::{model::events as lava_events, prelude::*};
use poise::serenity_prelude as serenity;
use songbird::SerenityInit;
use tracing::info;

pub struct Data {
    pub lavalink: LavalinkClient,
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
            event_handler: |ctx, event, _framework_ctx, _data|  {
                Box::pin(async move {
                    events::message_event(&ctx, event).await;
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
                // poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                poise::builtins::register_in_guild(
                    ctx,
                    &framework.options().commands,
                    guild,
                ).await?;

                let events = lava_events::Events {
                    raw: Some(events::raw_event),
                    ready: Some(events::ready_event),
                    track_start: Some(events::track_start),
                    track_end: Some(events::track_end),
                    ..Default::default()
                };

                let node_local = NodeBuilder {
                    hostname: lavalink_host,
                    is_ssl: lavalink_ssl,
                    events: lava_events::Events::default(),
                    password: lavalink_password,
                    user_id: ctx.cache.current_user().id.into(),
                    session_id: None,
                };

                let client = LavalinkClient::new(
                    events,
                    vec![node_local],
                    NodeDistributionStrategy::round_robin(),
                )
                .await;

                Ok(Data { lavalink: client })
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
