use std::sync::Arc;
use axum::{
    routing::post,
    Router,
};
use crate::audio::handlers::{transcribe_audio, translate_audio};
use crate::audio::models::WhisperState;

pub fn audio_router(whisper_state: Arc<WhisperState>) -> Router {
    Router::new()
        .route("/transcriptions", post(transcribe_audio))
        .route("/translations", post(translate_audio))
        .with_state(whisper_state)
}
