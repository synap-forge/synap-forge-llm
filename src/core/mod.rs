pub mod generator;
pub mod load_model;
pub mod output_stream;
pub mod medgemma;

use candle_transformers::models::gemma3::Model as Gemma;
use candle_transformers::models::llama::Llama;
use candle_transformers::models::mistral::Model as Mistral;
use crate::core::medgemma::Model as MedGemmaModel;

use candle_transformers::models::gemma3::Config as GemmaConfig;
use candle_transformers::models::llama::Config as LlamaConfig;
use candle_transformers::models::mistral::Config as MistralConfig;
use crate::core::medgemma::Config as MedGemmaConfig;


#[derive(Clone)]
pub enum Model {
    Llama(Llama),
    Mistral(Mistral),
    Gemma(Gemma),
    MedGemma(MedGemmaModel),
}

impl Model {
    pub fn clear_kv_cache(&mut self) {
        match self {
            // The Llama model's cache is managed externally in the generator loop.
            Model::Llama(_) => {},
            Model::Gemma(model) => model.clear_kv_cache(),
            Model::MedGemma(model) => model.clear_kv_cache(),
            // Mistral does not have a cache to clear
            Model::Mistral(_) => {},
        }
    }
}

#[derive(Clone)]
pub enum Config {
    Llama(LlamaConfig),
    Mistral(MistralConfig),
    Gemma(GemmaConfig),
    MedGemma(MedGemmaConfig),
}

impl Config {
    pub fn get_model_name(&self) -> &'static str {
        match self {
            Config::Llama(_) => "llama",
            Config::Mistral(_) => "mistral",
            Config::Gemma(_) => "gemma",
            Config::MedGemma(_) => "medgemma",
           }
    }
}

use candle_core::DType;

pub static USE_FLASH_ATTN: bool = false;

// Enum to represent different model types
#[derive(Clone, Debug)]
pub enum ModelType {
    Llama,
    Mistral,
    Gemma,
    MedGemma,
}

// Enum to represent different model configurations
#[derive(Clone, Debug)]
pub enum ModelConfig {
    Llama {
        model_name: String,
        model_revision: String,
        dtype: Option<DType>,
    },
    Mistral {
        model_name: String,
        model_revision: String,
        dtype: Option<DType>,
    },
    Gemma {
        model_name: String,
        model_revision: String,
        model_type: ModelType,
        dtype: Option<DType>,
    },
    MedGemma {
        model_name: String,
        model_revision: String,
        dtype: Option<DType>,
    },
}

impl ModelConfig {
    pub fn default_llama3() -> Self {
        ModelConfig::Llama {
            model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            model_revision: "0e9e39f249a16976918f6564b8830bc894c89659".to_string(),
            dtype: Some(DType::F16),
        }
    }

    pub fn init(model_type: ModelType) -> Self {
        match model_type {
            ModelType::Llama => ModelConfig::Llama {
                model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                model_revision: "0e9e39f249a16976918f6564b8830bc894c89659".to_string(),
                dtype: Some(DType::F16),
            },
            ModelType::Mistral => ModelConfig::Mistral {
                model_name: "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
                model_revision: "2dcff66eac0c01dc50e4c41eea959968232187fe".to_string(),
                dtype: Some(DType::F16),
            },
            ModelType::Gemma => ModelConfig::Gemma {
                model_name: "google/gemma-3-4b-it".to_string(),
                model_revision: "093f9f388b31de276ce2de164bdc2081324b9767".to_string(),
                model_type: ModelType::Gemma,
                dtype: Some(DType::F16),
            },
            ModelType::MedGemma => ModelConfig::MedGemma {
                model_name: "google/medgemma-4b-it".to_string(),
                model_revision: "698f7911b8e0569ff4ebac5d5552f02a9553063c".to_string(),
                dtype: Some(DType::F16),
            },
        }
    }

    pub fn from_huggingface(
        model_name: &str, 
        model_type: ModelType, 
        model_revision: Option<&str>, 
        dtype: Option<DType>) -> Self {
        match model_type {
            ModelType::Llama => ModelConfig::Llama {
                model_name: model_name.to_string(),
                model_revision: model_revision.unwrap_or("main").to_string(),
                dtype,
            },
            ModelType::Mistral => ModelConfig::Mistral {
                model_name: model_name.to_string(),
                model_revision: model_revision.unwrap_or("main").to_string(),
                dtype,
            },
            ModelType::Gemma => ModelConfig::Gemma {
                model_name: model_name.to_string(),
                model_revision: model_revision.unwrap_or("main").to_string(),
                model_type,
                dtype,
            },
            ModelType::MedGemma => ModelConfig::MedGemma {
                model_name: model_name.to_string(),
                model_revision: model_revision.unwrap_or("main").to_string(),
                dtype,
            },
        }
    }

    pub fn get_model_name(&self) -> &str {
        match self {
            ModelConfig::Llama { model_name, .. } => model_name,
            ModelConfig::Mistral { model_name, .. } => model_name,
            ModelConfig::Gemma { model_name, .. } => model_name,
            ModelConfig::MedGemma { model_name, .. } => model_name,
        }
    }

    pub fn get_model_revision(&self) -> Option<&str> {
        match self {
            ModelConfig::Llama { model_revision, .. } => Some(model_revision),
            ModelConfig::Mistral { model_revision, .. } => Some(model_revision),
            ModelConfig::Gemma { model_revision, .. } => Some(model_revision),
            ModelConfig::MedGemma { model_revision, .. } => Some(model_revision),
        }
    }

    pub fn get_model_type(&self) -> ModelType {
        match self {
            ModelConfig::Llama { .. } => ModelType::Llama,
            ModelConfig::Mistral { .. } => ModelType::Mistral,
            ModelConfig::Gemma { model_type, .. } => model_type.clone(),
            ModelConfig::MedGemma { .. } => ModelType::MedGemma,
        }
    }

    pub fn get_dtype(&self) -> Option<DType> {
        match self {
            ModelConfig::Llama { dtype, .. } => *dtype,
            ModelConfig::Mistral { dtype, .. } => *dtype,
            ModelConfig::Gemma { dtype, .. } => *dtype,
            ModelConfig::MedGemma { dtype, .. } => *dtype,
        }
    }
}
