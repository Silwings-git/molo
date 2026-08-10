//! Structured output example: typed output (run_typed) and hand-written
//! schema forms.
//!
//! Run: `cargo run --example structured` (uses FakeProvider, no real API
//! needed). Real-endpoint integration is in the comment at the end
//! (OpenAI-compatible, including the transport-mode fallback switch).

use molo::{Agent, FakeProvider, FakeReply, ReActAgent};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// The target output type: `schemars` derives generate the JSON Schema
/// (regenerated automatically on every run_typed), and `serde` derives handle
/// deserialization — the same pipeline as tool argument schemas.
#[derive(Debug, Deserialize, JsonSchema)]
struct Weather {
    city: String,
    #[schemars(description = "Temperature in Celsius")]
    temperature: f64,
    #[schemars(description = "Weather condition")]
    condition: String,
}

#[tokio::main]
async fn main() -> Result<(), molo::AgentError> {
    // Form 1: typed output — the return type is declared in the let annotation, and this run
    // generates the schema from Weather automatically; on validation failure the error is fed
    // back to the model for retry (separate budget, 3 attempts by default).
    let mut agent = ReActAgent::new(
        FakeProvider::new([FakeReply::Text(
            r#"{"city":"Beijing","temperature":31.5,"condition":"Sunny"}"#.into(),
        )]),
        molo::tool::ToolRegistry::new(),
        "You are a weather assistant; output JSON only",
    );

    let weather: Weather = agent
        .run_typed("How is the weather in Beijing today")
        .await?;
    println!(
        "typed output: city = {}, temperature = {}°C, condition = {}",
        weather.city, weather.temperature, weather.condition
    );

    // Form 2: hand-written schema — run returns JSON text, and structural
    // constraints are still validated framework-side (replies not matching the
    // schema are fed back to the model for retry).
    let mut agent = ReActAgent::new(
        FakeProvider::new([FakeReply::Text(r#"{"city":"Shanghai"}"#.into())]),
        molo::tool::ToolRegistry::new(),
        "You are a weather assistant; output JSON only",
    )
    .with_structured_output(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
    }));

    let answer = agent.run("How is the weather in Shanghai").await?;
    println!("hand-written schema: {answer}");

    // Form 3: real endpoints (OpenAI / DeepSeek / Zhipu / Moonshot / Ollama compatible).
    // The transport mode is configurable: Native (default, strict json_schema mode) /
    // JsonObject (fallback, only constrains "is JSON") / Off (no response_format sent) —
    // structural consistency is always guaranteed by framework-side validation.
    //
    // let provider = molo::OpenAiProvider::new(
    //     "https://api.openai.com/v1",
    //     "sk-...",
    //     "gpt-4o-mini",
    // )
    // .with_structured_output_mode(molo::StructuredOutputMode::Native);
    // let mut agent = ReActAgent::new(provider, molo::tool::ToolRegistry::new(),
    //     "You are a weather assistant; output JSON only");
    // let weather: Weather = agent.run_typed("How is the weather in Beijing today").await?;

    Ok(())
}
