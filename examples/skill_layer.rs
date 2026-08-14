use molo::{
    LoadSkillReferenceTool, RunContext, SharedState, Skill, SkillLayer, SkillRegistry,
    SkillResourceStore, ToolRegistry,
};
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Arc::new(SkillRegistry::new());
    registry.add(Skill::parse(
        "---\n\
         name: code-review\n\
         description: Review code changes for bugs and regressions\n\
         allowed-tools: read_file search_repo\n\
         ---\n\
         Read the diff first, then inspect changed call sites.",
    )?);

    let layer = SkillLayer::new(Arc::clone(&registry));
    let assembly = layer.assemble();

    let mut tools = ToolRegistry::new();
    if let Some(load_skill) = assembly.load_skill_tool {
        tools.register_with_source(load_skill, layer.load_skill_source())?;
    }
    tools.register_with_source(
        LoadSkillReferenceTool::new(
            Arc::clone(&registry),
            None,
            layer.activation_state(),
            SkillResourceStore::default(),
        ),
        LoadSkillReferenceTool::source("skills", molo::ToolTrustLevel::Project),
    )?;

    println!("prompt fragment:\n{}", assembly.prompt_fragment);
    println!("tools: {:?}", tools.names());

    let loaded = tools
        .call_named(
            "load_skill",
            r#"{"name":"code-review"}"#,
            &RunContext::new("skill-layer-example"),
            &SharedState::new(),
        )
        .await?;
    println!("loaded skill bytes: {}", loaded.to_string().len());
    Ok(())
}
