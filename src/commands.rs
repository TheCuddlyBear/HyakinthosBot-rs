use crate::{Context, Error};
use futures::{future, stream::StreamExt};
use lavalink_rs::prelude::*;
use poise::serenity_prelude::{self as serenity, ChannelId, Http, Mentionable};

/// Helper to join the user's voice channel if not already connected.
async fn ensure_joined(
    ctx: &Context<'_>,
    guild_id: serenity::GuildId,
) -> Result<PlayerContext, Error> {
    let lava_client = ctx.data().lavalink.clone();

    if let Some(player) = lava_client.get_player_context(guild_id) {
        return Ok(player);
    }

    let user_channel_id = {
        let guild = ctx
            .guild()
            .ok_or("Could not find guild in cache. Make sure this command is run in a server.")?;
        guild
            .voice_states
            .get(&ctx.author().id)
            .and_then(|voice_state| voice_state.channel_id)
    };

    let connect_to = match user_channel_id {
        Some(channel) => channel,
        None => {
            ctx.say("❌ You must be in a voice channel to use this command!")
                .await?;
            return Err("User not in a voice channel".into());
        }
    };

    let manager = songbird::get(ctx.serenity_context())
        .await
        .ok_or("Songbird Voice client not initialized")?
        .clone();

    match manager.join_gateway(guild_id, connect_to).await {
        Ok((connection_info, _)) => {
            let player = lava_client
                .create_player_context_with_data::<(ChannelId, std::sync::Arc<Http>)>(
                    guild_id,
                    connection_info,
                    std::sync::Arc::new((ctx.channel_id(), ctx.serenity_context().http.clone())),
                )
                .await?;

            ctx.say(format!("🔊 Joined {}", connect_to.mention())).await?;
            Ok(player)
        }
        Err(why) => {
            ctx.say(format!("❌ Error joining the voice channel: {}", why))
                .await?;
            Err(why.into())
        }
    }
}

/// Helper to enqueue loaded tracks and start playback if currently idle.
async fn enqueue_and_play(
    ctx: &Context<'_>,
    player: &PlayerContext,
    loaded_tracks: lavalink_rs::model::track::Track,
    custom_title: Option<String>,
) -> Result<(), Error> {
    let mut playlist_info = None;

    let mut tracks: Vec<TrackInQueue> = match loaded_tracks.data {
        Some(TrackLoadData::Track(x)) => vec![x.into()],
        Some(TrackLoadData::Search(x)) => {
            if x.is_empty() {
                ctx.say("🔍 No matching tracks found.").await?;
                return Ok(());
            }
            vec![x[0].clone().into()]
        }
        Some(TrackLoadData::Playlist(x)) => {
            playlist_info = Some(x.info);
            x.tracks.iter().map(|x| x.clone().into()).collect()
        }
        Some(TrackLoadData::Error(err)) => {
            ctx.say(format!("❌ Error loading track: {}", err.message))
                .await?;
            return Ok(());
        }
        None => {
            ctx.say("🔍 No tracks found.").await?;
            return Ok(());
        }
    };

    if tracks.is_empty() {
        ctx.say("🔍 No playable tracks found.").await?;
        return Ok(());
    }

    if let Some(title) = custom_title {
        if let Some(first) = tracks.first_mut() {
            if first.track.info.title == "Unknown title" || first.track.info.title.is_empty() {
                first.track.info.title = title;
            }
        }
    }

    if let Some(info) = playlist_info {
        ctx.say(format!(
            "➕ Added playlist to queue: **{}** ({} tracks)",
            info.name,
            tracks.len()
        ))
        .await?;
    } else {
        let track = &tracks[0].track;
        if let Some(uri) = &track.info.uri {
            if uri.starts_with("http") {
                ctx.say(format!(
                    "➕ Added to queue: [**{} - {}**](<{}>)",
                    track.info.author, track.info.title, uri
                ))
                .await?;
            } else {
                ctx.say(format!(
                    "➕ Added to queue: **{} - {}**",
                    track.info.author, track.info.title
                ))
                .await?;
            }
        } else {
            ctx.say(format!(
                "➕ Added to queue: **{} - {}**",
                track.info.author, track.info.title
            ))
            .await?;
        }
    }

    for i in &mut tracks {
        i.track.user_data = Some(serde_json::json!({
            "requester_id": ctx.author().id.get()
        }));
    }

    let queue = player.get_queue();
    queue.append(tracks.into())?;

    if let Ok(player_data) = player.get_player().await {
        if player_data.track.is_none() && queue.get_track(0).await.is_ok_and(|x| x.is_some()) {
            player.skip()?;
        }
    }

    Ok(())
}

/// Play a song from a URL or search query.
#[poise::command(slash_command, prefix_command, guild_only)]
pub async fn play(
    ctx: Context<'_>,
    #[description = "Search query or URL to play"]
    #[rest]
    query: Option<String>,
) -> Result<(), Error> {
    ctx.defer().await?;

    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let Ok(player) = ensure_joined(&ctx, guild_id).await else {
        return Ok(());
    };

    let lava_client = ctx.data().lavalink.clone();

    let search_query = if let Some(term) = query {
        if term.starts_with("http://") || term.starts_with("https://") {
            term
        } else {
            SearchEngines::YouTube.to_query(&term)?
        }
    } else {
        // If no query was provided, resume or start playing from the queue
        if let Ok(player_data) = player.get_player().await {
            let queue = player.get_queue();
            if player_data.paused {
                player.set_pause(false).await?;
                ctx.say("▶️ Resumed playback.").await?;
            } else if player_data.track.is_none()
                && queue.get_track(0).await.is_ok_and(|x| x.is_some())
            {
                player.skip()?;
                ctx.say("▶️ Starting queue playback.").await?;
            } else {
                ctx.say("ℹ️ Please provide a search query or URL to play.")
                    .await?;
            }
        }
        return Ok(());
    };

    let loaded_tracks = lava_client.load_tracks(guild_id, &search_query).await?;
    enqueue_and_play(&ctx, &player, loaded_tracks, None).await?;

    Ok(())
}

/// Play an uploaded audio or video attachment via Lavalink.
#[poise::command(slash_command, prefix_command, guild_only, rename = "play-local")]
pub async fn play_local(
    ctx: Context<'_>,
    #[description = "Audio or video file attachment to play"] file: serenity::Attachment,
) -> Result<(), Error> {
    ctx.defer().await?;

    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let Ok(player) = ensure_joined(&ctx, guild_id).await else {
        return Ok(());
    };

    let lava_client = ctx.data().lavalink.clone();
    let filename = file.filename.clone();

    // Try loading directly from Discord's attachment URL first (Lavalink HTTP source),
    // and also download to a local file in case Lavalink is configured for local files.
    let mut loaded_tracks = lava_client.load_tracks(guild_id, &file.url).await?;

    if loaded_tracks.data.is_none()
        || matches!(loaded_tracks.data, Some(TrackLoadData::Error(_)))
    {
        let download_dir = std::env::temp_dir().join("rust_discord_music_uploads");
        tokio::fs::create_dir_all(&download_dir).await?;

        let safe_name = filename
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();

        let local_path = download_dir.join(format!("{}_{}", file.id.get(), safe_name));
        let bytes = file.download().await?;
        tokio::fs::write(&local_path, bytes).await?;

        let local_path_str = local_path.to_string_lossy().to_string();
        loaded_tracks = lava_client.load_tracks(guild_id, &local_path_str).await?;
    }

    enqueue_and_play(&ctx, &player, loaded_tracks, Some(filename)).await?;

    Ok(())
}

/// Skip the currently playing song.
#[poise::command(slash_command, prefix_command, guild_only)]
pub async fn skip(ctx: Context<'_>) -> Result<(), Error> {
    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let lava_client = ctx.data().lavalink.clone();

    let Some(player) = lava_client.get_player_context(guild_id) else {
        ctx.say("❌ The bot is not connected to a voice channel.")
            .await?;
        return Ok(());
    };

    let now_playing = player.get_player().await?.track;

    if let Some(np) = now_playing {
        player.skip()?;
        ctx.say(format!("⏭️ Skipped **{}**", np.info.title)).await?;
    } else {
        ctx.say("ℹ️ Nothing is currently playing.").await?;
    }

    Ok(())
}

/// View the current music queue.
#[poise::command(slash_command, prefix_command, guild_only)]
pub async fn queue(ctx: Context<'_>) -> Result<(), Error> {
    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let lava_client = ctx.data().lavalink.clone();

    let Some(player) = lava_client.get_player_context(guild_id) else {
        ctx.say("❌ The bot is not connected to a voice channel.")
            .await?;
        return Ok(());
    };

    let queue = player.get_queue();
    let player_data = player.get_player().await?;

    let total_count = queue.get_count().await?;
    let max = total_count.min(10);

    let queue_message = queue
        .enumerate()
        .take_while(|(idx, _)| future::ready(*idx < max))
        .map(|(idx, x)| {
            let requester = x
                .track
                .user_data
                .as_ref()
                .and_then(|d| d.get("requester_id"))
                .and_then(|v| v.as_u64())
                .map(|id| format!(" | Requested by <@{}>", id))
                .unwrap_or_default();

            if let Some(uri) = &x.track.info.uri {
                if uri.starts_with("http") {
                    format!(
                        "`{}.` [**{} - {}**](<{}>){}",
                        idx + 1,
                        x.track.info.author,
                        x.track.info.title,
                        uri,
                        requester
                    )
                } else {
                    format!(
                        "`{}.` **{} - {}**{}",
                        idx + 1,
                        x.track.info.author,
                        x.track.info.title,
                        requester
                    )
                }
            } else {
                format!(
                    "`{}.` **{} - {}**{}",
                    idx + 1,
                    x.track.info.author,
                    x.track.info.title,
                    requester
                )
            }
        })
        .collect::<Vec<_>>()
        .await
        .join("\n");

    let now_playing_message = if let Some(track) = player_data.track {
        let time_s = player_data.state.position / 1000 % 60;
        let time_m = player_data.state.position / 1000 / 60;
        let time = format!("{:02}:{:02}", time_m, time_s);

        let requester = track
            .user_data
            .as_ref()
            .and_then(|d| d.get("requester_id"))
            .and_then(|v| v.as_u64())
            .map(|id| format!(", Requested by <@{}>", id))
            .unwrap_or_default();

        let status_icon = if player_data.paused { "⏸️" } else { "🎶" };

        if let Some(uri) = &track.info.uri {
            if uri.starts_with("http") {
                format!(
                    "{} **Now Playing:** [**{} - {}**](<{}>) | `{}`{}",
                    status_icon, track.info.author, track.info.title, uri, time, requester
                )
            } else {
                format!(
                    "{} **Now Playing:** **{} - {}** | `{}`{}",
                    status_icon, track.info.author, track.info.title, time, requester
                )
            }
        } else {
            format!(
                "{} **Now Playing:** **{} - {}** | `{}`{}",
                status_icon, track.info.author, track.info.title, time, requester
            )
        }
    } else {
        "🔇 **Now Playing:** Nothing".to_string()
    };

    let footer = if total_count > max {
        format!("\n\n*...and {} more track(s) in queue.*", total_count - max)
    } else if total_count == 0 {
        "\n\n*The queue is currently empty.*".to_string()
    } else {
        String::new()
    };

    let response = if queue_message.is_empty() {
        format!("{}{}", now_playing_message, footer)
    } else {
        format!(
            "{}\n\n**Up Next:**\n{}{}",
            now_playing_message, queue_message, footer
        )
    };

    ctx.say(response).await?;

    Ok(())
}

/// Stop playback and clear the queue.
#[poise::command(slash_command, prefix_command, guild_only)]
pub async fn stop(ctx: Context<'_>) -> Result<(), Error> {
    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let lava_client = ctx.data().lavalink.clone();

    let Some(player) = lava_client.get_player_context(guild_id) else {
        ctx.say("❌ The bot is not connected to a voice channel.")
            .await?;
        return Ok(());
    };

    player.get_queue().clear()?;
    player.stop_now().await?;

    ctx.say("⏹️ Stopped playback and cleared the queue.")
        .await?;

    Ok(())
}

/// Set the playback volume (0 - 1000%).
#[poise::command(slash_command, prefix_command, guild_only)]
pub async fn volume(
    ctx: Context<'_>,
    #[description = "Volume level between 0 and 1000 (default 100)"]
    #[min = 0]
    #[max = 1000]
    level: u16,
) -> Result<(), Error> {
    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let lava_client = ctx.data().lavalink.clone();

    let Some(player) = lava_client.get_player_context(guild_id) else {
        ctx.say("❌ The bot is not connected to a voice channel.")
            .await?;
        return Ok(());
    };

    let clamped = level.min(1000);
    player.set_volume(clamped).await?;

    ctx.say(format!("🔊 Volume set to **{}%**", clamped))
        .await?;

    Ok(())
}

/// Pause or toggle pause on the currently playing song.
#[poise::command(slash_command, prefix_command, guild_only)]
pub async fn pause(
    ctx: Context<'_>,
    #[description = "Optional: true to pause, false to resume (toggles if omitted)"] state: Option<
        bool,
    >,
) -> Result<(), Error> {
    let guild_id = ctx.guild_id().ok_or("Command must be used in a server")?;
    let lava_client = ctx.data().lavalink.clone();

    let Some(player) = lava_client.get_player_context(guild_id) else {
        ctx.say("❌ The bot is not connected to a voice channel.")
            .await?;
        return Ok(());
    };

    let player_data = player.get_player().await?;
    let new_pause_state = state.unwrap_or(!player_data.paused);

    player.set_pause(new_pause_state).await?;

    if new_pause_state {
        ctx.say("⏸️ Paused playback.").await?;
    } else {
        ctx.say("▶️ Resumed playback.").await?;
    }

    Ok(())
}
