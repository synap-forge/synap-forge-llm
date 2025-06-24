use std::time::{Duration, Instant};
use anyhow::Result;
use axum::body::Bytes;
use axum::extract::MatchedPath;
use axum::http::{HeaderMap, Request};
use axum::response::Response;
use axum::Router;
use axum::routing::{get, post};
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::trace::TraceLayer;
use tracing::{debug, debug_span, error, Span};
use tracing::log::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use synap_forge_llm::core::load_model::initialise_model;
use synap_forge_llm::core::{ModelConfig, ModelType};
use synap_forge_llm::openai::http_service::{create_chat_completion, create_completion, create_embedding, delete_model, health, list_models, retrieve_model};
use synap_forge_llm::openai::models::AppState;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                "synap_forge_llm=debug,tower_http=debug,axum::rejection=trace".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let Ok(api_token) = std::env::var("HF_TOKEN") else {
        panic!("HF_TOKEN environment variable is not set");
    };

    info!("Model is loading in memory");
    let before = Instant::now();

    let app_state: AppState;

    #[cfg(any(feature = "metal", feature = "default"))]
    {
        let model_config = ModelConfig::init(ModelType::MedGemma);
        app_state = initialise_model(api_token, Some(model_config))
            .expect("Failed to initialize model");
    }
        let after = Instant::now();

    // let whisper_state = Arc::new(WhisperState::new("openai/whisper-tiny".to_string(), "main".to_string())?);

    info!(
        "Model loaded and is ready now with Elapsed time: {:.2?}",
        Duration::from_secs(after.duration_since(before).as_millis() as u64)
    );

    let openai_router = Router::new()
        .route("/health", get(health))
        .route("/chat/completions", post(create_chat_completion))
        .route("/completions", post(create_completion))
        .route("/embeddings", post(create_embedding))
        .route("/models", get(list_models))
        .route(
            "/models/{model_id}",
            get(retrieve_model).delete(delete_model),
        )
        .with_state(app_state);

    let app = Router::new()
        .nest("/v1", openai_router)
                // .nest("/v1/audio", audio_router(whisper_state))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<_>| {
                    let matched_path = request
                        .extensions()
                        .get::<MatchedPath>()
                        .map(MatchedPath::as_str);

                    debug_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri(),
                        matched_path = matched_path,
                        version = ?request.version(),
                        headers = ?request.headers(),
                    )
                })
                .on_request(|request: &Request<_>, _span: &Span| {
                    debug!(
                        "Started {} request to {} body {:?}",
                        request.method(),
                        request.uri(),
                        request.body()
                    );
                })
                .on_response(|response: &Response, latency: Duration, _span: &Span| {
                    debug!(
                        "Response with status {} latency {} ms",
                        response.status(),
                        latency.as_millis()
                    );
                })
                .on_body_chunk(|chunk: &Bytes, _latency: Duration, _span: &Span| {
                    debug!("Sending {} bytes", chunk.len());
                })
                .on_failure(
                    |error: ServerErrorsFailureClass, _latency: Duration, _span: &Span| {
                        error!("Something went wrong: {}", error);
                    },
                )
                .on_eos(
                    |trailers: Option<&HeaderMap>, stream_duration: Duration, _span: &Span| {
                        debug!(
                            "Stream completed in {:?}, trailers: {:?}",
                            stream_duration, trailers
                        );
                    },
                ),
        );

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    
    info!("🚀 Server listening on 0.0.0.0:3000");

    axum::serve(tcp_listener, app).await.unwrap();

    Ok(())
}
