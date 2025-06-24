use crate::core::output_stream::TokenOutputStream;
use crate::core::Model;
use crate::openai::models::AppState;

use candle_core::{DType, Device, Tensor, IndexOp};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use candle_transformers::models::llama::Cache as LlamaCache;

use tokenizers::Tokenizer;
use tracing::trace;

/// A struct representing text generation using various models.
///
/// The `TextGeneration` struct contains fields for the model, tokenizer, logits processor, repeat penalty, repeat last n, and configuration.
/// It provides methods to create a new `TextGeneration` instance and generate text based on a given prompt.
pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f64,
    repeat_last_n: usize,
    config: crate::core::Config,
}

impl TextGeneration {
    /// Creates a new `TextGeneration` instance with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for text generation.
    /// * `tokenizer` - The tokenizer to use for encoding and decoding text.
    /// * `seed` - The seed value for the random number generator.
    /// * `temperature` - Optional temperature value for sampling.
    /// * `top_p` - Optional top-p value for nucleus sampling.
    /// * `top_k` - Optional top-k value for nucleus sampling.
    /// * `repeat_penalty` - The repeat penalty value.
    /// * `repeat_last_n` - The number of last tokens to consider for repeat penalty.
    /// * `device` - The device to use for computations.
    ///
    /// # Returns
    ///
    /// A new `TextGeneration` instance with the specified parameters.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: i64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f64,
        repeat_last_n: usize,
        device: &Device,
        config: crate::core::Config,
    ) -> Self {
        let logits_processor = {
            let temperature = temperature.unwrap_or_else(|| 0f64);

            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed as u64, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            config,
        }
    }

    /// Generates text based on the given prompt and maximum number of tokens.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt string to use for text generation.
    /// * `max_tokens` - Optional maximum number of tokens to generate.
    ///
    /// # Returns
    ///
    /// The generated text as a string.
    pub(crate) fn generate(
        &mut self,
        prompt: String,
        max_tokens: Option<i32>,
        d_type: DType,
    ) -> (String, i32) {
        self.model.clear_kv_cache();
        self.tokenizer.clear();
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        trace!("Got tokens!");

        let eos_token = match &self.model {
            Model::Llama(_) => self.tokenizer.tokenizer().token_to_id("</s>"),
            Model::Mistral(_) => self.tokenizer.tokenizer().token_to_id("</s>"),
            Model::Gemma(_) => self.tokenizer.tokenizer().token_to_id("<eos>"),
            Model::MedGemma(_) => self.tokenizer.tokenizer().token_to_id("<eos>"),
        }
        .unwrap();

        let mut tokens = tokens;
        let mut text_generated = String::new();
        let mut token_generated = 0;
        let mut index_pos = 0;

        let mut llama_cache = if let (Model::Llama(_), crate::core::Config::Llama(config)) = (&self.model, &self.config) {
            Some(LlamaCache::new(true, d_type, config, &self.device).unwrap())
        } else {
            None
        };

        for index in 0..max_tokens.unwrap_or(1024) {
            let (logits, ctxt_len) = match &mut self.model {
                Model::Llama(model) => {
                    let cache = llama_cache.as_mut().unwrap();
                    let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                        (1, index_pos)
                    } else {
                        (tokens.len(), 0)
                    };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device).unwrap().unsqueeze(0).unwrap();
                    let logits = model.forward(&input, context_index, cache).unwrap();
                    (logits, ctxt.len())
                }
                Model::Mistral(model) => {
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device).unwrap().unsqueeze(0).unwrap();
                    let logits = model.forward(&input, index_pos).unwrap();
                    (logits, ctxt.len())
                }
                Model::Gemma(model) => {
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device).unwrap().unsqueeze(0).unwrap();
                    let logits = model.forward(&input, index_pos).unwrap();
                    (logits, ctxt.len())
                }
                Model::MedGemma(model) => {
                    let context_size = if index > 0 { 1 } else { tokens.len() };
                    let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                    let input = Tensor::new(ctxt, &self.device).unwrap().unsqueeze(0).unwrap();
                    let logits = model.forward(&input, index_pos).unwrap();
                    (logits, ctxt.len())
                }
            };

            let logits = logits.to_dtype(DType::F32).unwrap();

            index_pos += ctxt_len;
            let logits = logits.squeeze(0).unwrap();

            let seq_len = logits.dim(0).unwrap();
            let logits = logits.i(seq_len - 1).unwrap();

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                let logits = candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty as f32,
                    &tokens[start_at..],
                )
                .unwrap();
                logits
            };

            match self.logits_processor.sample(&logits) {
                Ok(next_token) => {
                    token_generated += 1;
                    tokens.push(next_token);

                    if next_token == eos_token {
                        break;
                    }
                    if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                        text_generated.push_str(&t);
                    }
                }
                Err(e) => {
                    tracing::error!("Error sampling logits: {}", e);
                    break;
                }
            }
        }
        (text_generated, token_generated)
    }
}

impl
    From<(
        AppState,
        Option<f64>,
        Option<f64>,
        Option<usize>,
        Option<i64>,
        Option<f64>,
        Option<usize>,
    )> for TextGeneration
{
    fn from(
        tuple: (
            AppState,
            Option<f64>,
            Option<f64>,
            Option<usize>,
            Option<i64>,
            Option<f64>,
            Option<usize>,
        ),
    ) -> Self {
        let (app_state, temperature, top_p, top_k, seed, repeat_penalty, repeat_last_n) = tuple;

        Self::new(
            app_state.model,
            app_state.tokenizer,
            seed.unwrap_or(299792458),        // seed RNG
            temperature,                         // temperature
            top_p,                               // top_p - Nucleus sampling probability stuff
            top_k,                               // top_k - Nucleus sampling probability stuff
            repeat_penalty.unwrap_or(1.1),     // repeat penalty (repeat_penalty)
            repeat_last_n.unwrap_or(64),       // context size to consider for the repeat penalty
            &app_state.device,
            app_state.config,
        )
    }
}
