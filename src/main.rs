use anyhow::Result;

// #[tokio::main]
// async fn main() -> Result<()> {
//     tracing_subscriber::registry()
//         .with(
//             tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
//                 // axum logs rejections from built-in extractors with the `axum::rejection`
//                 // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
//                 "synap_forge_llm=debug,tower_http=debug,axum::rejection=trace".into()
//             }),
//         )
//         .with(tracing_subscriber::fmt::layer())
//         .init();
//
//     let Ok(api_token) = std::env::var("HF_TOKEN") else {
//         return Err(anyhow::anyhow!("Error getting HF_TOKEN env var"));
//     };
//
//     info!("Model is loading in memory");
//     let before = Instant::now();
//
//     let state: AppState;
//
//     #[cfg(any(feature = "metal", feature = "default"))]
//     {
//         state = initialise_model(api_token)?;
//     }
//     let after = Instant::now();
//
//     info!(
//         "Model loaded and is ready now with Elapsed time: {:.2?}",
//         Duration::from_secs(after.duration_since(before).as_millis() as u64)
//     );
//
//     let openai_router = Router::new()
//         .route("/health", get(health))
//         .route("/chat/completions", post(create_chat_completion))
//         .route("/completions", post(create_completion))
//         .route("/embeddings", post(create_embedding))
//         .route("/models", get(list_models))
//         .route(
//             "/models/:model_id",
//             get(retrieve_model).delete(delete_model),
//         )
//         .with_state(state)
//         .layer(
//             TraceLayer::new_for_http()
//                 .make_span_with(|request: &Request<_>| {
//                     // Create span with request details
//                     let matched_path = request
//                         .extensions()
//                         .get::<MatchedPath>()
//                         .map(MatchedPath::as_str);
//
//                     debug_span!(
//                         "http_request",
//                         method = %request.method(),
//                         uri = %request.uri(),
//                         matched_path = matched_path,
//                         version = ?request.version(),
//                         headers = ?request.headers(),
//                     )
//                 })
//                 .on_request(|request: &Request<_>, _span: &Span| {
//                     // Log when request starts
//                     debug!(
//                         "Started {} request to {} body {:?}",
//                         request.method(),
//                         request.uri(),
//                         request.body()
//                     );
//                 })
//                 .on_response(|response: &Response, latency: Duration, _span: &Span| {
//                     // Log response details
//                     debug!(
//                         "Response completed with body {:?} status {} in {:?}",
//                         response.body(),
//                         response.status(),
//                         latency
//                     );
//                 })
//                 .on_body_chunk(|chunk: &Bytes, latency: Duration, _span: &Span| {
//                     // Log body chunk details
//                     debug!(
//                         "Sent body chunk of size {} bytes after {:?}",
//                         chunk.len(),
//                         latency
//                     );
//                 })
//                 .on_eos(
//                     |trailers: Option<&HeaderMap>, stream_duration: Duration, _span: &Span| {
//                         // Log end of stream
//                         debug!(
//                             "Stream completed in {:?}, trailers: {:?}",
//                             stream_duration, trailers
//                         );
//                     },
//                 )
//                 .on_failure(
//                     |error: ServerErrorsFailureClass, latency: Duration, _span: &Span| {
//                         // Log errors
//                         error!("Request failed after {:?}: {:?}", latency, error);
//                     },
//                 ),
//         );
//
//     let main_router = Router::new().nest("/v1", openai_router);
//
//     let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
//
//     axum::serve(tcp_listener, main_router).await.unwrap();
//
//     Ok(())
// }

use clap::Parser;
use hound::WavReader;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use ct2rs::Whisper;

/// Transcribe a file using Whisper models.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory that contains model.bin.
    model_dir: PathBuf,
    /// Path to the WAVE file.
    audio_file: PathBuf,
}

/// Entry point for the command-line interface.
///
/// This function parses the command-line arguments using the `clap` crate, creates a `Whisper` instance,
/// reads the given audio file, runs the transcription using the `generate` method, and prints the
/// results to stdout.
///
/// The function returns a `Result` to handle any potential errors that may occur.
#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse();

    let whisper = Whisper::new(args.model_dir, Default::default())?;

    let samples = read_audio(args.audio_file, whisper.sampling_rate())?;

    let res = whisper.generate(&samples, None, false, &Default::default())?;
    for r in res {
        println!("{}", r);
    }

    Ok(())
}

fn read_audio<T: AsRef<Path>>(path: T, sample_rate: usize) -> Result<Vec<f32>> {
    // Should use a better resampling algorithm.
    fn resample(samples: Vec<f32>, src_rate: usize, target_rate: usize) -> Vec<f32> {
        samples
            .into_iter()
            .step_by(src_rate / target_rate)
            .collect()
    }

    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let max = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
    let samples = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect::<Vec<f32>>();

    if spec.channels == 1 {
        return Ok(resample(samples, spec.sample_rate as usize, sample_rate));
    }

    let mut mono = vec![];
    for chunk in samples.chunks(2) {
        if chunk.len() == 2 {
            mono.push((chunk[0] + chunk[1]) / 2.);
        }
    }

    Ok(resample(mono, spec.sample_rate as usize, sample_rate))
}

#[cfg(not(feature = "cuda"))]
fn main() {}
