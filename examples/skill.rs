//! Skill example: capability packs — the full mechanism from SKILL.md to an
//! activatable state.
//!
//! A skill = a directory containing `SKILL.md` (YAML frontmatter + Markdown
//! body), optionally with `references/` / `scripts/` / `assets/` resources;
//! it follows the Agent Skills open protocol. The core mechanism is
//! **progressive disclosure**: the model initially sees only one line per
//! skill (a menu of name + description); when a task matches a description, it
//! reads the body by name via the
//! [`load_skill`](molo::skill::LoadSkillTool) tool. Skills do not package
//! tools — tools stay in the ToolRegistry, and a skill declares its
//! dependencies with `allowed-tools`.
//!
//! # Assembly options
//!
//! - [`with_skills`](molo::agent::ReActAgent::with_skills): dynamic
//!   progressive disclosure (the protocol's primary shape) — the menu is
//!   merged into the system prompt and load_skill is registered into the
//!   ToolRegistry;
//! - [`activate_skill`](molo::agent::ReActAgent::activate_skill): explicit
//!   user activation — the body is merged into the system prompt without the
//!   model's decision (application-level parsing of slash commands);
//! - [`SkillRegistry::from_dir`](molo::skill::SkillRegistry::from_dir):
//!   directory discovery — scans the skills directory, tolerantly skipping
//!   broken skills.
//!
//! This example is self-contained (driven by FakeProvider), needs no API key,
//! just run:
//! `cargo run --example skill`

use molo::agent::{Agent, ReActAgent};
use molo::provider::{FakeProvider, FakeReply};
use molo::skill::{Skill, SkillRegistry};
use molo::{Message, ToolCall, ToolRegistry};
use std::sync::Arc;

/// Shared FakeProvider wrapper: request history stays accessible after the
/// Agent runs (FakeProvider does not implement Clone, so wrap it in an Arc).
#[derive(Clone)]
struct SharedFake(Arc<FakeProvider>);

impl SharedFake {
    /// A snapshot of the request history (delegating to FakeProvider).
    fn requests(&self) -> Vec<molo::ChatRequest> {
        self.0.requests()
    }
}

#[molo::async_trait]
impl molo::Provider for SharedFake {
    async fn chat(
        &self,
        request: molo::ChatRequest,
    ) -> Result<molo::ChatResponse, molo::ProviderError> {
        self.0.chat(request).await
    }

    async fn stream_chat(
        &self,
        request: molo::ChatRequest,
    ) -> Result<
        futures::stream::BoxStream<'static, Result<molo::StreamEvent, molo::ProviderError>>,
        molo::ProviderError,
    > {
        self.0.stream_chat(request).await
    }
}

/// Skill 1: code review (loaded through the progressive-disclosure path).
const CODE_REVIEW_SKILL: &str = r#"---
name: code-review
description: Review code changes against team conventions and output a list of issues sorted by severity
allowed-tools:
  - Bash(git:*)
---
Review process:
1. Read the scope of the changes and the diff;
2. Check each file for issues: correctness / naming / documentation;
3. Output the list of issues sorted by severity.
"#;

/// Skill 2: release notes generation (loaded through the pre-activation path).
const RELEASE_NOTES_SKILL: &str = r#"---
name: release-notes
description: Generate release notes from the commit history
---
Rules:
- Group entries into "New features / Fixes / Maintenance";
- One short line per entry, no details.
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "== Path 1: progressive disclosure (menu always present + load_skill reads on demand) =="
    );

    // 1. Parse SKILL.md (pure in-memory, synchronous); registry add is the basis for hot plugging.
    let registry = SkillRegistry::new();
    registry.add(Skill::parse(CODE_REVIEW_SKILL)?);
    registry.add(Skill::parse(RELEASE_NOTES_SKILL)?);

    // 2. Assembly: the menu is merged into the system prompt, and load_skill is registered into the ToolRegistry.
    //    Script: the model calls load_skill for code-review in the first round, then answers in the second.
    let fake = SharedFake(Arc::new(FakeProvider::new([
        FakeReply::ToolCalls {
            content: "the task matches the code-review skill, load it first".into(),
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "load_skill".into(),
                arguments: r#"{"name":"code-review"}"#.into(),
            }],
        },
        FakeReply::Text("Reviewed per the skill process; found 2 issues.".into()),
    ])));
    let mut agent = ReActAgent::new(
        fake.clone(),
        ToolRegistry::new(),
        "You are a code review assistant",
    )
    .with_skills(registry);

    let answer = agent.run("review the changes under docs/").await?;
    println!("answer: {answer}");

    // 3. Observe the requests: the first round's System message carries the menu; the second round's ToolResult carries the skill body.
    let requests = fake.requests();
    if let Message::System(system) = &requests[0].messages[0] {
        println!("-- skill menu in the first round's system prompt:");
        for line in system.lines().filter(|l| l.starts_with("- ")) {
            println!("   {line}");
        }
    }
    for message in &requests[1].messages {
        if let Message::ToolResult { content, .. } = message {
            println!("-- skill body returned by load_skill (excerpt):");
            for line in content.lines().take(2) {
                println!("   {line}");
            }
        }
    }

    println!();
    println!(
        "== Path 2: explicit user activation (activate_skill; the body is not decided by the model) =="
    );

    let registry = SkillRegistry::new();
    registry.add(Skill::parse(RELEASE_NOTES_SKILL)?);
    let fake = SharedFake(Arc::new(FakeProvider::new([FakeReply::Text(
        "OK, generated the release notes per the rules.".into(),
    )])));
    let mut agent = ReActAgent::new(
        fake.clone(),
        ToolRegistry::new(),
        "You are a release assistant",
    )
    .with_skills(registry);
    // The user / application layer activates directly: the body is merged into the system prompt immediately; the model does not need to call load_skill.
    assert!(agent.activate_skill("release-notes"));
    let answer = agent.run("generate the release notes").await?;
    println!("answer: {answer}");

    if let Message::System(system) = &fake.requests()[0].messages[0] {
        println!("-- system prompt contains the skill body directly (pre-activated section):");
        for line in system.lines().filter(|l| l.starts_with("[")) {
            println!("   {line}");
        }
    }

    println!();
    println!(
        "== Path 3: multi-source directory discovery (from_dirs, user directory + project directory merged) =="
    );

    // Self-contained demo: write two sets of skill files into a temp directory, then scan across sources.
    let dir = std::env::temp_dir().join(format!("molo-skill-demo-{}", std::process::id()));
    let user_dir = dir.join("user-skills");
    let project_dir = dir.join("project-skills");
    let _ = std::fs::remove_dir_all(&dir);
    // User level: code-review (user version) + a generic skill.
    std::fs::create_dir_all(user_dir.join("code-review")).unwrap();
    std::fs::write(user_dir.join("code-review/SKILL.md"), CODE_REVIEW_SKILL).unwrap();
    std::fs::create_dir_all(user_dir.join("translate")).unwrap();
    std::fs::write(
        user_dir.join("translate/SKILL.md"),
        "---\nname: translate\ndescription: Translate text\n---\nRules: keep the original meaning and formatting.",
    )
    .unwrap();
    // Project level: a project version of code-review (overrides the user version) + a broken skill (skipped).
    std::fs::create_dir_all(project_dir.join("code-review")).unwrap();
    std::fs::write(
        project_dir.join("code-review/SKILL.md"),
        "---\nname: code-review\ndescription: Review code changes against team conventions\n---\nProject-specific review steps.",
    )
    .unwrap();
    std::fs::create_dir_all(project_dir.join("bad-skill")).unwrap();
    std::fs::write(
        project_dir.join("bad-skill/SKILL.md"),
        "---\nname: BadSkill\ndescription: Invalid skill name\n---\n",
    )
    .unwrap();

    // Multi-source merge: the project level comes later, so the same-named code-review overrides the user level; broken skills are skipped.
    let discovered = SkillRegistry::from_dirs(&[user_dir, project_dir]).await;
    println!(
        "discovered {} skills (same-name merged, broken skills skipped):",
        discovered.skills().len()
    );
    for skill in discovered.skills() {
        println!("   - {}: {}", skill.name(), skill.description());
    }
    let code_review = discovered
        .get("code-review")
        .expect("code-review must exist");
    println!("-- code-review body comes from the project level (overriding the user level):");
    for line in code_review.body().lines().take(2) {
        println!("   {line}");
    }
    let _ = std::fs::remove_dir_all(&dir);

    Ok(())
}
