use anyhow::Result;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::Config;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

use crate::audio::decode_whispers;

pub struct WhisperState {
    pub model: Arc<Mutex<decode_whispers::model::Model>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: Config,
    pub mel_filters: Vec<f32>,
    pub device: Device,
}

impl WhisperState {
    pub fn new(model_id: String, revision: String) -> Result<Self> {
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let (config_filename, tokenizer_filename, weights_filename, mel_filters_filename) = decode_whispers::hub::load_files(&repo)?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
        let mel_filters = decode_whispers::hub::load_mel_filters(&mel_filters_filename)?;
        let d_type = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], d_type, &device)?
        };
        let whisper_model = crate::audio::decode_whispers::model::Model::Normal(
            crate::audio::decode_whispers::model::m::model::Whisper::load(&vb, config.clone())?,
        );

        Ok(Self {
            model: Arc::new(Mutex::new(whisper_model)),
            tokenizer: Arc::new(tokenizer),
            config,
            mel_filters,
            device,
        })
    }
}


#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampGranularity {
    Word,
    Segment,
}

#[derive(Debug, Clone)]
pub struct CreateTranscriptionRequest {
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Vec<TimestampGranularity>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTranscriptionResponseJson {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Segment {
    pub id: usize,
    pub seek: f64,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTranscriptionResponseVerboseJson {
    pub task: String,
    pub language: String,
    pub duration: f64,
    pub text: String,
    pub words: Option<Vec<Word>>,
    pub segments: Vec<Segment>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum CreateTranscriptionResponse {
    Json(CreateTranscriptionResponseJson),
    VerboseJson(CreateTranscriptionResponseVerboseJson),
}

#[derive(Debug, Clone)]
pub struct CreateTranslationRequest {
    pub model: String,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTranslationResponseJson {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTranslationResponseVerboseJson {
    pub task: String,
    pub language: String,
    pub duration: f64,
    pub text: String,
    pub segments: Vec<Segment>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum CreateTranslationResponse {
    Json(CreateTranslationResponseJson),
    VerboseJson(CreateTranslationResponseVerboseJson),
}
