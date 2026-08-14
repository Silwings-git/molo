//! Step-wise agent action protocol shared by runtimes and harnesses.

use crate::effect::{EffectObservation, EffectRequest};
use crate::provider::{ChatRequest, ChatResponse};
use crate::run::{RunMetadata, RunOutput};

/// Provider request emitted by a step-wise agent kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRequest {
    /// Model request id, unique within the run or agent implementation.
    pub id: String,
    /// Provider chat request to execute.
    pub chat: ChatRequest,
    /// Framework/application metadata.
    pub metadata: RunMetadata,
}

impl ModelRequest {
    /// Constructs a model request.
    pub fn new(id: impl Into<String>, chat: ChatRequest) -> Self {
        Self {
            id: id.into(),
            chat,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets framework/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Successful provider response observed by a step-wise agent kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelObservation {
    /// Model request id matching [`ModelRequest::id`].
    pub request_id: String,
    /// Provider response.
    pub response: ChatResponse,
    /// Framework/application metadata.
    pub metadata: RunMetadata,
}

impl ModelObservation {
    /// Constructs a model observation.
    pub fn new(request_id: impl Into<String>, response: ChatResponse) -> Self {
        Self {
            request_id: request_id.into(),
            response,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets framework/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Observation fed back to a step-wise agent kernel.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Observation {
    /// Successful model response.
    Model(ModelObservation),
    /// Effect observation produced by an outer harness.
    Effect(EffectObservation),
    /// Batch effect observations produced by an outer harness.
    ///
    /// This is the companion to [`AgentAction::RequestEffects`]. The outer
    /// runtime may execute the requests sequentially or in parallel, then
    /// feeds the completed set back to the kernel as one step.
    Effects(Vec<EffectObservation>),
}

/// Next action requested by a step-wise agent kernel.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AgentAction {
    /// The run is complete.
    Respond {
        /// Final structured run output.
        output: RunOutput,
    },
    /// The outer runtime should execute a provider request and feed back
    /// [`Observation::Model`].
    RequestModel {
        /// Provider request.
        request: ModelRequest,
    },
    /// The outer runtime should govern and execute an effect request, then
    /// feed back [`Observation::Effect`].
    RequestEffect {
        /// Effect request.
        request: EffectRequest,
    },
    /// The outer runtime should govern and execute multiple effect requests,
    /// then feed back [`Observation::Effects`].
    RequestEffects {
        /// Effect requests.
        requests: Vec<EffectRequest>,
    },
}
