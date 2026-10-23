use lavalink_rs::{hook, model::events, prelude::*};
use poise::serenity_prelude::{ChannelId, Http};
use tracing::{debug, info};


#[hook]
pub async fn raw_event(_: LavalinkClient, session_id: String, event: &serde_json::Value) {
    if event["op"].as_str() == Some("event") || event["op"].as_str() == Some("playerUpdate") {
        debug!("{:?} -> {:?}", session_id, event);
    }
}

#[hook]
pub async fn ready_event(client: LavalinkClient, session_id: String, event: &events::Ready) {
    let _ = client.delete_all_player_contexts().await;
    info!("Lavalink node ready: {:?} -> {:?}", session_id, event);
}

#[hook]
pub async fn track_start(client: LavalinkClient, _session_id: String, event: &events::TrackStart) {
    let Some(player_context) = client.get_player_context(event.guild_id) else {
        return;
    };
    let Ok(data) = player_context.data::<(ChannelId, std::sync::Arc<Http>)>() else {
        return;
    };
    let (channel_id, http) = (&data.0, &data.1);

    let track = &event.track;
    let requester = track
        .user_data
        .as_ref()
        .and_then(|d| d.get("requester_id"))
        .and_then(|v| v.as_u64())
        .map(|id| format!(" | Requested by <@{}>", id))
        .unwrap_or_default();

    let msg = if let Some(uri) = &track.info.uri {
        if uri.starts_with("http") {
            format!(
                "🎶 Now playing: [**{} - {}**](<{}>){}",
                track.info.author, track.info.title, uri, requester
            )
        } else {
            format!(
                "🎶 Now playing: **{} - {}**{}",
                track.info.author, track.info.title, requester
            )
        }
    } else {
        format!(
            "🎶 Now playing: **{} - {}**{}",
            track.info.author, track.info.title, requester
        )
    };

    let _ = channel_id.say(http, msg).await;
}

#[hook]
pub async fn track_end(client: LavalinkClient, _session_id: String, event: &events::TrackEnd) {
    if let Some(player_context) = client.get_player_context(event.guild_id) {
        debug!(
            "Track ended. Songs left in queue: {:?}",
            player_context.get_queue().get_count().await
        );
    }
}
