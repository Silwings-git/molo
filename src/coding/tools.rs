use crate::RiskLevel;
use crate::tool::{
    SideEffectLevel, Tool, ToolContext, ToolError, ToolPolicy, ToolResult, ToolSchema,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::git::{GitOperation, GitStatusRequest};
use super::payload::{
    ApplyPatchPayload, CommandPayload, GitPayload, ListFilesPayload, ReadFilePayload, SearchPayload,
};

/// Model-visible adapter that requests a governed workspace file read.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadFileTool;

#[crate::async_trait]
impl Tool for ReadFileTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "read_file",
            "Read a file from the governed workspace",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Root-relative workspace path" },
                    "max_bytes": { "type": "integer", "minimum": 1 }
                },
                "required": ["path"]
            }),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::ReadOnly,
            risk: RiskLevel::Low,
            ..ToolPolicy::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        effect_from_payload::<ReadFilePayload>(arguments, context, ReadFilePayload::into_effect)
    }
}

/// Model-visible adapter that requests a governed workspace listing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListFilesTool;

#[crate::async_trait]
impl Tool for ListFilesTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "list_files",
            "List files from the governed workspace",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Root-relative workspace path" },
                    "recursive": { "type": "boolean" },
                    "max_entries": { "type": "integer", "minimum": 1 },
                    "include_hidden": { "type": "boolean" },
                    "respect_gitignore": { "type": "boolean" }
                },
                "required": ["path", "recursive"]
            }),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::ReadOnly,
            risk: RiskLevel::Low,
            ..ToolPolicy::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        effect_from_payload::<ListFilesPayload>(arguments, context, ListFilesPayload::into_effect)
    }
}

/// Model-visible adapter that requests governed repository search.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchRepoTool;

#[crate::async_trait]
impl Tool for SearchRepoTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "search_repo",
            "Search text in the governed workspace",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "max_matches": { "type": "integer", "minimum": 1 },
                    "context_lines": { "type": "integer", "minimum": 0 }
                },
                "required": ["query"]
            }),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::ReadOnly,
            risk: RiskLevel::Low,
            ..ToolPolicy::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        effect_from_payload::<SearchPayload>(arguments, context, SearchPayload::into_effect)
    }
}

/// Model-visible adapter that requests a governed structured patch.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApplyPatchTool;

#[crate::async_trait]
impl Tool for ApplyPatchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "apply_patch",
            "Apply a structured patch through the governed workspace",
            json!({
                "type": "object",
                "properties": {
                    "patch": { "type": "object" },
                    "expected_versions": { "type": "array" },
                    "dry_run": { "type": "boolean" }
                },
                "required": ["patch", "dry_run"]
            }),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::Write,
            risk: RiskLevel::Medium,
            requires_confirmation: true,
            ..ToolPolicy::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        effect_from_payload::<ApplyPatchPayload>(arguments, context, ApplyPatchPayload::into_effect)
    }
}

/// Model-visible adapter that requests governed command execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunCommandTool;

#[crate::async_trait]
impl Tool for RunCommandTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "run_command",
            "Run an explicit argv command through the governed command executor",
            json!({
                "type": "object",
                "properties": {
                    "request": { "type": "object" }
                },
                "required": ["request"]
            }),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::External,
            risk: RiskLevel::Medium,
            requires_confirmation: true,
            ..ToolPolicy::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        effect_from_payload::<CommandPayload>(arguments, context, CommandPayload::into_effect)
    }
}

/// Model-visible adapter that requests read-only git status.
#[derive(Debug, Clone, Copy, Default)]
pub struct GitStatusTool;

#[crate::async_trait]
impl Tool for GitStatusTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "git_status",
            "Inspect git status without mutating git state",
            json!({
                "type": "object",
                "properties": {
                    "include_branch": { "type": "boolean" }
                }
            }),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::ReadOnly,
            risk: RiskLevel::Low,
            ..ToolPolicy::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: GitStatusArgs = serde_json::from_value(arguments).map_err(ToolError::from)?;
        let effect = GitPayload {
            operation: GitOperation::Status(GitStatusRequest {
                include_branch: args.include_branch.unwrap_or(true),
            }),
        }
        .into_effect()
        .map_err(|error| ToolError::InvalidArguments(error.to_string()))?
        .with_source(context.tool_call_id, context.tool_name);
        Ok(ToolResult::Effect(effect))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct GitStatusArgs {
    include_branch: Option<bool>,
}

fn effect_from_payload<P>(
    arguments: serde_json::Value,
    context: ToolContext<'_>,
    into_effect: impl FnOnce(P) -> Result<crate::EffectRequest, super::CodingError>,
) -> Result<ToolResult, ToolError>
where
    P: for<'de> Deserialize<'de>,
{
    let payload: P = serde_json::from_value(arguments).map_err(ToolError::from)?;
    let effect = into_effect(payload)
        .map_err(|error| ToolError::InvalidArguments(error.to_string()))?
        .with_source(context.tool_call_id, context.tool_name);
    Ok(ToolResult::Effect(effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RunContext, SharedState};

    #[tokio::test]
    async fn read_file_tool_returns_effect() {
        let run = RunContext::new("tool");
        let state = SharedState::new();
        let context = ToolContext::new(&run, &state, "call-1", "read_file");
        let result = ReadFileTool
            .call(json!({"path": "Cargo.toml", "max_bytes": 10}), context)
            .await
            .unwrap();
        let ToolResult::Effect(effect) = result else {
            panic!("expected effect");
        };
        assert_eq!(effect.source.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(effect.source.tool_name.as_deref(), Some("read_file"));
    }

    #[tokio::test]
    async fn read_file_tool_rejects_invalid_json_arguments() {
        let run = RunContext::new("tool");
        let state = SharedState::new();
        let context = ToolContext::new(&run, &state, "call-1", "read_file");
        let err = ReadFileTool
            .call(json!({"path": "../secret"}), context)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }
}
