use anyhow::Result;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use candle_core::Tensor;
use candle_transformers::models::whisper::audio::pcm_to_mel;
use serde_json::json;
use std::io::Write;
use std::sync::Arc;
use tempfile::NamedTempFile;
use tracing::info;

use crate::audio::models::{
    CreateTranscriptionResponse, CreateTranscriptionResponseJson, CreateTranslationResponse,
    CreateTranslationResponseJson, WhisperState,
};
use crate::audio::decode_whispers::model;
use crate::audio::decode_whispers::multilingual;
use crate::audio::decode_whispers::pcm_decode::pcm_decode;

async fn process_multipart_request(
    mut multipart: Multipart,
) -> Result<(NamedTempFile, String), Response> {
    let mut temp_file = NamedTempFile::new().map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to create temporary file: {}", err) })),
        )
            .into_response()
    })?;
    let mut model = "decode_whispers-1".to_string();

    while let Some(field) = multipart.next_field().await.map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response()
    })? {
        let name = if let Some(name) = field.name() {
            name.to_string()
        } else {
            continue;
        };

        let data = field.bytes().await.map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response()
        })?;

        match name.as_str() {
            "file" => {
                temp_file.write_all(&data).map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": format!("Failed to write to temporary file: {}", err) })),
                    )
                        .into_response()
                })?;
            }
            "model" => {
                model = String::from_utf8(data.to_vec()).map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(json!({ "error": err.to_string() })),
                    )
                        .into_response()
                })?;
            }
            _ => {}
        }
    }

    Ok((temp_file, model))
}

pub async fn transcribe_audio(
    State(whisper_state): State<Arc<WhisperState>>,
    multipart: Multipart,
) -> Result<Json<CreateTranscriptionResponse>, Response> {
    let (temp_file, model_name) = process_multipart_request(multipart).await?;
    let (pcm_data, _sample_rate) = pcm_decode(temp_file.path()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to decode audio: {}", e),
        )
            .into_response()
    })?;

        let mel = pcm_to_mel(
        &whisper_state.config,
        &pcm_data,
        &whisper_state.mel_filters,
    );
    let mel_len = mel.len();
    let mel_frames = mel_len / whisper_state.config.num_mel_bins;
    let mel = Tensor::from_vec(
        mel,
        (1, whisper_state.config.num_mel_bins, mel_frames),
        &whisper_state.device,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create mel tensor: {}", e),
        )
            .into_response()
    })?;

    let mut whisper_model = whisper_state.model.lock().await;

    let language_token = if model_name.is_empty() {
        Some(
            multilingual::detect_language(&mut *whisper_model, &*whisper_state.tokenizer, &mel)
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to detect language: {}", e),
                    )
                        .into_response()
                })?,
        )
    } else {
        None
    };

    let mut decoder = model::Decoder::new(
        whisper_model.clone(),
        (*whisper_state.tokenizer).clone(),
        299792458, // seed
        &whisper_state.device,
        language_token,
        Some(model::Task::Transcribe),
        false, // timestamps
        false, // verbose_timestamps
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create decoder: {}", e),
        )
            .into_response()
    })?;

    let segments = decoder.run(&mel).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to transcribe audio: {}", e),
        )
            .into_response()
    })?;

    let transcription = segments
        .iter()
        .map(|s| s.dr.text.as_str())
        .collect::<String>();

    let response = CreateTranscriptionResponseJson {
        text: transcription,
    };

    Ok(Json(CreateTranscriptionResponse::Json(response)))
}

pub async fn translate_audio(
    State(whisper_state): State<Arc<WhisperState>>,
    multipart: Multipart,
) -> Result<Json<CreateTranslationResponse>, Response> {
    info!("Translating audio");
    let (temp_file, _) = process_multipart_request(multipart).await?;

    let (pcm_data, _sample_rate) = pcm_decode(temp_file.path()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to decode audio: {}", e),
        )
            .into_response()
    })?;

        let mel = pcm_to_mel(
        &whisper_state.config,
        &pcm_data,
        &whisper_state.mel_filters,
    );
    let mel_len = mel.len();
    let mel_frames = mel_len / whisper_state.config.num_mel_bins;
    let mel = Tensor::from_vec(
        mel,
        (1, whisper_state.config.num_mel_bins, mel_frames),
        &whisper_state.device,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create mel tensor: {}", e),
        )
            .into_response()
    })?;

    let mut whisper_model = whisper_state.model.lock().await;

    let language_token = Some(
        multilingual::detect_language(&mut *whisper_model, &*whisper_state.tokenizer, &mel)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to detect language: {}", e),
                )
                    .into_response()
            })?,
    );

    let mut decoder = model::Decoder::new(
        whisper_model.clone(),
        (*whisper_state.tokenizer).clone(),
        299792458, // Your seed
        &whisper_state.device,
        language_token,
        Some(model::Task::Translate),
        false, // Timestamps not supported for translation
        false, // verbose_timestamps
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create decoder: {}", e),
        )
            .into_response()
    })?;

    let segments = decoder.run(&mel).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to run translation: {}", e),
        )
            .into_response()
    })?;

    let response = CreateTranslationResponseJson {
        text: segments
            .iter()
            .map(|s| s.dr.text.as_str())
            .collect::<String>(),
    };

    Ok(Json(CreateTranslationResponse::Json(response)))
}
