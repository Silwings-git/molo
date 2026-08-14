use crate::args::ApprovalMode;
use crate::session::CliApprovalSummary;
use async_trait::async_trait;
use molo::{ApprovalBroker, ApprovalDecision, ApprovalError, ApprovalRequest, RunContext};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

/// CLI approval broker.
#[derive(Debug, Clone)]
pub struct CliApprovalBroker {
    mode: ApprovalMode,
    non_interactive: bool,
    summaries: Arc<Mutex<Vec<CliApprovalSummary>>>,
}

impl CliApprovalBroker {
    /// Constructs a broker.
    pub fn new(mode: ApprovalMode, non_interactive: bool) -> Self {
        Self {
            mode,
            non_interactive,
            summaries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns decisions captured by this broker.
    pub fn summaries(&self) -> Vec<CliApprovalSummary> {
        self.summaries
            .lock()
            .expect("CliApprovalBroker lock poisoned")
            .clone()
    }

    fn record(&self, request: &ApprovalRequest, decision: &ApprovalDecision) {
        let decision_text = match decision {
            ApprovalDecision::AllowOnce => "allow_once".to_string(),
            ApprovalDecision::AllowForSession => "allow_for_session".to_string(),
            ApprovalDecision::Deny { reason } => format!("deny: {reason}"),
            _ => "unknown".to_string(),
        };
        self.summaries
            .lock()
            .expect("CliApprovalBroker lock poisoned")
            .push(CliApprovalSummary {
                effect_id: request.effect_id.clone(),
                kind: format!("{:?}", request.kind),
                risk: format!("{:?}", request.risk),
                decision: decision_text,
                reason: request.reason.clone(),
            });
    }
}

#[async_trait]
impl ApprovalBroker for CliApprovalBroker {
    async fn approve(
        &self,
        request: ApprovalRequest,
        _context: &RunContext,
    ) -> Result<ApprovalDecision, ApprovalError> {
        let decision = match self.mode {
            ApprovalMode::Allow => ApprovalDecision::AllowOnce,
            ApprovalMode::Deny => ApprovalDecision::Deny {
                reason: "denied by CLI approval mode".to_string(),
            },
            ApprovalMode::Ask if self.non_interactive => ApprovalDecision::Deny {
                reason: "approval required in non-interactive mode".to_string(),
            },
            ApprovalMode::Ask => prompt_for_decision(&request)?,
        };
        self.record(&request, &decision);
        Ok(decision)
    }
}

fn prompt_for_decision(request: &ApprovalRequest) -> Result<ApprovalDecision, ApprovalError> {
    println!("approval required");
    println!("effect: {} {:?}", request.effect_id, request.kind);
    println!("risk: {:?}", request.risk);
    println!("reason: {}", request.reason);
    println!("description: {}", request.description);
    println!("payload: {}", request.payload_summary);
    println!("sandbox: {:?}", request.sandbox);
    println!("network: {:?}", request.network);
    print!("allow once [y], allow for session [s], deny [n]? ");
    io::stdout()
        .flush()
        .map_err(|error| ApprovalError::Broker(error.to_string()))?;
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .map_err(|error| ApprovalError::Broker(error.to_string()))?;
    Ok(match line.trim() {
        "y" | "yes" => ApprovalDecision::AllowOnce,
        "s" | "session" => ApprovalDecision::AllowForSession,
        _ => ApprovalDecision::Deny {
            reason: "denied by user".to_string(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molo::{EffectKind, NetworkPolicy, RiskLevel, SandboxPolicy};

    #[tokio::test]
    async fn non_interactive_ask_fails_closed() {
        let broker = CliApprovalBroker::new(ApprovalMode::Ask, true);
        let decision = broker
            .approve(
                ApprovalRequest {
                    run_id: "run".to_string(),
                    effect_id: "effect".to_string(),
                    kind: EffectKind::ExecuteCommand,
                    description: "run".to_string(),
                    risk: RiskLevel::High,
                    reason: "test".to_string(),
                    payload_summary: "{}".to_string(),
                    sandbox: SandboxPolicy::WorkspaceWrite,
                    network: NetworkPolicy::Deny,
                    metadata: Default::default(),
                },
                &RunContext::new("run"),
            )
            .await
            .unwrap();
        assert!(matches!(decision, ApprovalDecision::Deny { .. }));
    }
}
