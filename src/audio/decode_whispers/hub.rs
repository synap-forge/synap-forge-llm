use anyhow::Result;
use hf_hub::api::sync::ApiRepo;
use candle_core::{safetensors, Device};
use std::path::{Path, PathBuf};

pub fn load_files(repo: &ApiRepo) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let config = repo.get("config.json")?;
    let tokenizer = repo.get("tokenizer.json")?;
    let mel_filters = repo.get("mel_filters.safetensors")?;
    let weights = repo.get("model.safetensors")?;
    Ok((config, tokenizer, weights, mel_filters))
}

pub fn load_mel_filters<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<f32>> {
    let tensors = safetensors::load(path, &Device::Cpu)?;
    let tensor = tensors
        .get("mel_filters")
        .ok_or_else(|| anyhow::anyhow!("mel_filters tensor not found in safetensors file"))?;
    Ok(tensor.to_vec1()?)
}
