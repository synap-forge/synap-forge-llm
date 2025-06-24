use crate::core::output_stream::WeightMaps;
use crate::core::{Config, Model, ModelConfig, ModelType, USE_FLASH_ATTN};
use crate::openai::models::AppState;
use anyhow::Error as E;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Llama, LlamaConfig};
use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use candle_transformers::models::gemma3::{Config as GemmaConfig, Model as Gemma};

use crate::core::medgemma::{Model as MedGemmaModel, TopLevelConfig as MedGemmaTopLevelConfig};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Deserializer};
use serde_json::from_reader;
use std::collections::HashSet;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::info;

/// Loads SafeTensors weight files from a Hugging Face repository based on a JSON configuration.
///
/// This function reads a JSON file that contains a mapping of weight files, retrieves these files
/// from the specified repository, and returns their paths.
///
/// # Arguments
///
/// * `repo` - A reference to an `ApiRepo` instance representing the Hugging Face repository
/// * `json_file` - A string slice containing the path to the JSON configuration file within the repository
///
/// # Returns
///
/// * `anyhow::Result<Vec<std::path::PathBuf>>` - A Result containing a vector of PathBuf objects
///   pointing to the loaded weight files if successful, or an error if the operation fails
///
/// # Errors
///
/// This function will return an error if:
/// * The JSON file cannot be found in the repository
/// * The JSON file cannot be opened
/// * The JSON file contains invalid data that cannot be deserialized
/// * Any of the weight files specified in the JSON cannot be retrieved from the repository
///
/// # Example
///
/// ```rust,no_run
/// use hf_hub::api::sync::{ApiBuilder, ApiRepo};
///
/// let repo : ApiRepo;
/// let weight_files = synap_forge_llm::core::load_model::hub_load_safe_tensors(&repo, "model.safetensors.index.json")?;
/// ```
///
/// # Notes
///
/// The JSON file should contain a `weight_map` field that specifies the paths to the SafeTensors
/// weight files within the repository. The function assumes that all paths in the weight map are valid
/// and accessible within the repository.
pub fn hub_load_safe_tensors(repo: &ApiRepo, json_file: &str) -> anyhow::Result<Vec<PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: WeightMaps = from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let path_bufs: Vec<PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(path_bufs)
}

/// Deserializes a JSON object into a `HashSet<String>`.
///
/// This function takes a deserializer and attempts to deserialize it into a
/// `HashSet<String>`. It expects the input to be a JSON object where the values
/// are strings. If the input is not an object, it returns an error.
///
/// # Parameters
///
/// - `deserializer`: A deserializer that implements the `Deserializer` trait,
///   which is used to read the JSON data.
///
/// # Returns
///
/// Returns a result containing either:
/// - `Ok(HashSet<String>)`: A set of strings extracted from the values of the
///   JSON object.
/// - `Err(D::Error)`: An error if the input is not a valid JSON object or if
///   deserialization fails.
///
/// # Errors
///
/// Returns an error if the input is not a JSON object or if any other
/// deserialization error occurs.
pub fn deserialize_weight_map<'de, D>(deserializer: D) -> anyhow::Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

/// Retrieves a `Tokenizer` from a specified repository.
///
/// This function attempts to load a `Tokenizer` by first fetching the filename
/// of the tokenizer configuration from the provided `ApiRepo`. It then reads
/// the tokenizer data from the specified file.
///
/// # Parameters
///
/// - `repo`: A reference to an `ApiRepo` instance, which is used to access
///   the tokenizer configuration file.
///
/// # Returns
///
/// Returns a result containing either:
/// - `Ok(Tokenizer)`: The loaded `Tokenizer` instance if successful.
/// - `Err(anyhow::Error)`: An error if the filename cannot be retrieved or
///   if loading the tokenizer from the file fails.
///
/// # Errors
///
/// This function may return an error if:
/// - The tokenizer filename cannot be obtained from the repository.
/// - There is an issue reading the tokenizer data from the file.
fn get_tokenizer(repo: &ApiRepo) -> anyhow::Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;

    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

/// Retrieves a configuration from a specified repository.
///
/// This function attempts to load a configuration by first fetching the filename
/// of the configuration file from the provided `ApiRepo`. It then reads the
/// configuration data from the specified file and converts it into a `Box<dyn std::any::Any>` instance.
///
/// # Parameters
///
/// - `repo`: A reference to an `ApiRepo` instance, which is used to access
///   the configuration file.
/// - `model_type`: The type of the model.
///
/// # Returns
///
/// Returns a result containing either:
/// - `Ok(Box<dyn std::any::Any>)`: The loaded configuration instance if successful.
/// - `Err(anyhow::Error)`: An error if the filename cannot be retrieved,
///   if reading the configuration file fails, or if deserialization fails.
///
/// # Errors
///
/// This function may return an error if:
/// - The configuration filename cannot be obtained from the repository.
/// - There is an issue reading the configuration data from the file.
/// - Deserialization of the configuration data fails.


/// Retrieves the preferred computational device.
///
/// This function attempts to create a computational device by first trying to
/// initialize a CUDA device. If that fails, it then tries to initialize a
/// Metal device. If both CUDA and Metal devices are unavailable, it defaults
/// to using a CPU device.
///
/// # Returns
///
/// Returns a `Device` instance representing the selected computational device.
/// The function will return:
/// - A CUDA device if available.
/// - A Metal device if CUDA is not available.
/// - A CPU device if neither CUDA nor Metal devices are available.
fn get_device() -> Device {
    let device_cuda = Device::new_cuda(0);
    let device_metal = Device::new_metal(0);

    let device = device_cuda.or(device_metal).unwrap_or(Device::Cpu);

    info!("Device Info {:?}", device);

    device
}

/// Initializes a machine learning model and its associated components.
///
/// This function sets up the application state by retrieving the necessary
/// resources, including the model repository, tokenizer, device, and
/// configuration. It loads the model from safe tensor files and prepares
/// it for use.
///
/// # Parameters
///
/// - `token`: A `String` representing the authentication token used to
///   access the model repository.
/// - `model_config`: A `ModelConfig` specifying the model to load
///
/// # Returns
///
/// Returns a result containing either:
/// - `Ok(AppState)`: The initialized application state containing the model,
pub fn initialise_model(token: String, model_config: Option<ModelConfig>) -> Result<AppState, E> {
    let model_config = model_config.unwrap_or_else(|| ModelConfig::default_llama3());
    
    let repo = {
        let model_name = model_config.get_model_name();
        let model_revision = model_config.get_model_revision().unwrap_or("main");
        
        let api = ApiBuilder::new()
            .with_token(Some(token))
            .build()?;
    
        api.repo(Repo::with_revision(model_name.to_string(), RepoType::Model, model_revision.to_string()))
    };

    let tokenizer = get_tokenizer(&repo)?;

    let device = get_device();
    let d_type = model_config.get_dtype().unwrap_or(DType::F16);

    let file_path = match hub_load_safe_tensors(&repo, "model.safetensors.index.json") {
        Ok(files) => files,
        Err(_) => {
            // Fetch the model.safetensors file directly
            vec![repo.get("model.safetensors")?]
        }
    };

    let config_path = repo.get("config.json")?;

    let (model, config) = match model_config.get_model_type() {
        ModelType::Llama => {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&file_path, d_type, &device)? };
            let config_data = std::fs::read(&config_path)?;
            let config: LlamaConfig = serde_json::from_slice(&config_data)?;
            let config = config.into_config(USE_FLASH_ATTN);
            let model = Model::Llama(Llama::load(vb, &config)?);
            (model, Config::Llama(config))
        },
        ModelType::Mistral => {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&file_path, d_type, &device)? };
            let config_data = std::fs::read(&config_path)?;
            let config: MistralConfig = serde_json::from_slice(&config_data)?;
            let model = Model::Mistral(Mistral::new(&config, vb)?);
            (model, Config::Mistral(config))
        },
        ModelType::Gemma => {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&file_path, d_type, &device)? };
            let config_data = std::fs::read(&config_path)?;
            let config: GemmaConfig = serde_json::from_slice(&config_data)?;
            let model = Model::Gemma(Gemma::new(USE_FLASH_ATTN, &config, vb)?);
            (model, Config::Gemma(config))
        },
        ModelType::MedGemma => {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&file_path, d_type, &device)? };
            let config_data = std::fs::read(&config_path)?;
            let top_level_config: MedGemmaTopLevelConfig = serde_json::from_slice(&config_data)?;
            let config = top_level_config.text_config;
            let model = Model::MedGemma(MedGemmaModel::new(&config, vb, USE_FLASH_ATTN)?);
            (model, Config::MedGemma(config))
        }
    };

    Ok((model, device, tokenizer, config, d_type).into())
}


