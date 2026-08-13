//! Skill: capability packages following the Agent Skills open protocol
//! (SKILL.md format).
//!
//! A skill is a directory containing `SKILL.md` (YAML frontmatter +
//! Markdown body) and optional `references/` / `scripts/` / `assets/`
//! resources; the frontmatter `name` / `description` decides when the
//! model triggers it. This module provides the full mechanism the protocol
//! defines: **format parsing → directory discovery → progressive
//! disclosure → activation** (the execution environment after activation
//! belongs to the application layer; this library does not run scripts).
//!
//! The core mechanism is **progressive disclosure**: the model initially
//! sees only a one-line `name + description` per skill (~100 tokens,
//! `menu()`); when a task matches the description, [`LoadSkillTool`] reads
//! the SKILL.md body by name and execution begins. Skills do not bundle
//! tools — tools stay in [`ToolRegistry`](crate::ToolRegistry), and skills
//! declare dependencies with `allowed-tools`.
//!
//! Companion assembly: [`ReActAgent::with_skills`](crate::agent::ReActAgent::with_skills)
//! merges the menu into the system prompt and registers [`LoadSkillTool`],
//! ready to use once assembled.
//!
//! # Example
//!
//! Parse a SKILL.md text (a self-contained skill, no resource directory):
//!
//! ```
//! use molo::skill::Skill;
//!
//! let skill = Skill::parse(
//!     "---\n\
//!      name: code-review\n\
//!      description: Review code changes against team conventions, find bugs and style issues\n\
//!      allowed-tools: Bash(git:*)\n\
//!      ---\n\
//!      Review steps: read the diff first, then check each file.",
//! )
//! .unwrap();
//!
//! assert_eq!(skill.name(), "code-review");
//! assert_eq!(skill.description(), "Review code changes against team conventions, find bugs and style issues");
//! assert_eq!(skill.body(), "Review steps: read the diff first, then check each file.");
//! assert_eq!(skill.allowed_tools()[0].name, "Bash");
//! ```

use indexmap::IndexMap;
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::io::AsyncReadExt;

use crate::tool::{
    Tool, ToolContext, ToolError, ToolMemoryPolicy, ToolOutput, ToolPolicy, ToolResult, ToolSchema,
};

/// SKILL.md read limit (bytes): prevents a malicious skill package from
/// blowing up memory at once.
const MAX_SKILL_FILE_BYTES: u64 = 1024 * 1024;

/// SKILL.md body limit (characters): the body is recorded as resident
/// context via protected tool output, so a single skill must not grow
/// unbounded.
const MAX_SKILL_BODY_CHARS: usize = 256 * 1024;

/// Tool dependencies declared by a skill: tool name + optional scope
/// (execution belongs to the application layer; this struct only parses
/// and matches).
///
/// Corresponds to the frontmatter `allowed-tools` field (experimental):
/// entries look like `Bash(git:*)` or `Python` — `name` is the tool name,
/// `scope` is the argument scope (e.g. `git:*` means any command under the
/// git namespace).
///
/// # Example
///
/// ```
/// use molo::skill::AllowedTool;
///
/// let bash_git = AllowedTool {
///     name: "Bash".into(),
///     scope: Some("git:*".into()),
/// };
/// // exact name match + scope prefix match (after stripping the trailing
/// // * wildcard)
/// assert!(bash_git.permits("Bash", "git:diff --stat"));
/// assert!(!bash_git.permits("Bash", "rm -rf /"));
/// assert!(!bash_git.permits("Python", "git:log"));
///
/// let python = AllowedTool { name: "Python".into(), scope: None };
/// // no scope: any arguments for the tool are allowed
/// assert!(python.permits("Python", "print('hello')"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedTool {
    /// Tool name (same as the tool registered in ToolRegistry).
    pub name: String,
    /// Argument scope: prefix-matching rule — when ending with `*`, the
    /// `*` stands for any suffix and the rest is prefix-matched against
    /// the argument text; `None` means arguments are unrestricted.
    pub scope: Option<String>,
}

impl AllowedTool {
    /// Whether a tool call falls within the scope this declaration allows.
    ///
    /// Pure function: the application layer (execution approval) plugs it
    /// into permission checks; this library does not reject on its own.
    ///
    /// # Panics
    ///
    /// Never panics.
    pub fn permits(&self, tool: &str, args: &str) -> bool {
        if self.name != tool {
            return false;
        }
        match &self.scope {
            None => true,
            Some(scope) => {
                // Scopes ending in `*` match by prefix after stripping the
                // wildcard.
                let prefix = scope.strip_suffix('*').unwrap_or(scope);
                args.starts_with(prefix)
            }
        }
    }
}

/// A parsed SKILL.md (a data packet, immutable; `Clone` copies by value).
///
/// Constructed by [`Skill::parse`] or [`Skill::from_dir`]; the difference
/// between the two sources is `base_dir` — the directory source carries
/// the skill root and supports reading resources via
/// [`load_reference`](Skill::load_reference); the text source has no
/// resource directory and returns [`SkillError::NotFound`] when reading.
///
/// Unknown top-level frontmatter fields (e.g. `user-invocable` carried by
/// ecosystem skills) do not error: they are stringified into
/// [`metadata`](Skill::metadata) for the reader to interpret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    name: String,
    description: String,
    body: String,
    license: Option<String>,
    compatibility: Option<String>,
    metadata: Vec<(String, String)>,
    allowed_tools: Vec<AllowedTool>,
    resources: Vec<PathBuf>,
    base_dir: Option<PathBuf>,
}

impl Skill {
    /// Parse SKILL.md text (pure in-memory operation, synchronous).
    ///
    /// frontmatter validation:
    /// - `name` required: 1-64 characters, kebab-case (lowercase letters /
    ///   digits / hyphens, not starting or ending with a hyphen, no
    ///   consecutive hyphens);
    /// - `description` required: non-empty, at most 1024 characters;
    /// - `compatibility` optional: at most 500 characters;
    /// - `allowed-tools` optional: supports a space-separated string form
    ///   (`Bash(git:*) Python`) and a YAML list form (`- Bash(git:*)`);
    /// - `metadata` optional: a key-value block with values stringified
    ///   (quotes stripped, numbers / booleans converted as-is to text);
    ///   nested structures are not supported and report
    ///   [`SkillError::InvalidFrontmatter`];
    /// - **unknown top-level fields are tolerated**: merged into metadata
    ///   (e.g. `user-invocable: true` → `("user-invocable", "true")`), so
    ///   ecosystem skills are not rejected.
    ///
    /// Body = everything after the end delimiter (a single newline right
    /// after the delimiter is stripped).
    ///
    /// # Errors
    ///
    /// All validation failures are described by [`SkillError`]; a missing
    /// or malformed `name` → [`SkillError::InvalidName`], a missing /
    /// empty / too-long `description` →
    /// [`SkillError::InvalidDescription`], any other frontmatter
    /// structural issue → [`SkillError::InvalidFrontmatter`].
    ///
    /// # Example
    ///
    /// ```
    /// use molo::skill::Skill;
    ///
    /// let skill = Skill::parse("---\nname: greet\ndescription: Say hello\n---\nHello!").unwrap();
    /// assert_eq!(skill.name(), "greet");
    ///
    /// // invalid name (uppercase letter): parsing fails
    /// assert!(Skill::parse("---\nname: Greet\ndescription: Say hello\n---\n").is_err());
    /// ```
    pub fn parse(content: &str) -> Result<Skill, SkillError> {
        let (fm, body) = parse_frontmatter(content)?;

        let name = fm
            .name
            .ok_or_else(|| SkillError::InvalidName("missing name field".into()))?;
        validate_name(&name)?;

        let description = fm
            .description
            .ok_or_else(|| SkillError::InvalidDescription("missing description field".into()))?;
        validate_description(&description)?;

        // Body limit: the body is recorded as protected resident context,
        // so reject when over the limit.
        if body.chars().count() > MAX_SKILL_BODY_CHARS {
            return Err(SkillError::InvalidBody(format!(
                "body exceeds size limit ({MAX_SKILL_BODY_CHARS} chars)"
            )));
        }

        Ok(Skill {
            name,
            description,
            body,
            license: fm.license,
            compatibility: fm.compatibility,
            metadata: fm.metadata,
            allowed_tools: fm.allowed_tools,
            resources: Vec::new(),
            base_dir: None,
        })
    }

    /// Load from a skill directory: read `SKILL.md` + verify the directory
    /// name matches the skill name + collect the resource list.
    ///
    /// Directory layout follows the protocol: the directory contains
    /// `SKILL.md`, and optional `references/` / `scripts/` / `assets/`
    /// subdirectories are collected recursively into
    /// [`resources`](Skill::resources) (relative paths, stable ordering).
    ///
    /// # Errors
    ///
    /// - no `SKILL.md` in the directory → [`SkillError::NotFound`];
    /// - parsing failures (see [`Skill::parse`]) propagate as-is;
    /// - directory name differs from the skill `name` →
    ///   [`SkillError::NameMismatch`];
    /// - filesystem errors → [`SkillError::Io`].
    pub async fn from_dir(path: &Path) -> Result<Skill, SkillError> {
        // Symlink defense: after resolution, SKILL.md must still be inside
        // the skill directory. The directory itself may be a symlink (e.g.
        // ~/skills → /mnt/skills); but if SKILL.md is a link to a file
        // outside the directory, its content would be read into the model
        // context, so it is rejected just like [`load_reference`].
        let dir = tokio::fs::canonicalize(path)
            .await
            .map_err(|e| SkillError::Io(format!("cannot resolve skill root: {e}")))?;
        let content = {
            let skill_md = match tokio::fs::canonicalize(dir.join("SKILL.md")).await {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Include the directory path so batch discovery
                    // failures can be located.
                    return Err(SkillError::NotFound(format!(
                        "no SKILL.md in directory: {}",
                        dir.display()
                    )));
                }
                Err(e) => return Err(e.into()),
            };
            if !skill_md.starts_with(&dir) {
                return Err(SkillError::NotFound("SKILL.md escapes skill root".into()));
            }
            // Read limit: reject when over the limit instead of reading
            // it all into memory (prevents malicious skill packages from
            // blowing up memory).
            let mut buf = String::new();
            let file = tokio::fs::File::open(&skill_md).await?;
            file.take(MAX_SKILL_FILE_BYTES + 1)
                .read_to_string(&mut buf)
                .await?;
            if buf.len() > MAX_SKILL_FILE_BYTES as usize {
                return Err(SkillError::InvalidBody(format!(
                    "SKILL.md exceeds size limit ({MAX_SKILL_FILE_BYTES} bytes)"
                )));
            }
            buf
        };

        let mut skill = Skill::parse(&content)?;
        let dir_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if skill.name != dir_name {
            return Err(SkillError::NameMismatch {
                name: skill.name.clone(),
                dir: dir_name,
            });
        }
        skill.resources = collect_resources(&dir).await;
        skill.base_dir = Some(dir);
        Ok(skill)
    }

    /// Skill name (kebab-case).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Skill description: one sentence of "what it does + when to use it",
    /// the basis on which the model decides whether to trigger it.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// SKILL.md body (the content shown to the model once triggered).
    pub fn body(&self) -> &str {
        &self.body
    }

    /// License (SPDX identifier, optional).
    pub fn license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    /// Compatibility note (optional, e.g. applicable frameworks and
    /// versions).
    pub fn compatibility(&self) -> Option<&str> {
        self.compatibility.as_deref()
    }

    /// Arbitrary metadata key-values (including unknown top-level fields,
    /// stringified; keeps frontmatter order).
    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    /// Declared tool dependencies (`allowed-tools`), which the application
    /// layer uses for execution approval.
    pub fn allowed_tools(&self) -> &[AllowedTool] {
        &self.allowed_tools
    }

    /// Resource file list (paths relative to the skill root; empty for
    /// text-parsed skills).
    pub fn resources(&self) -> &[PathBuf] {
        &self.resources
    }

    /// Skill root directory (`Some` for directory-loaded skills; `None`
    /// for text-parsed skills, which have no resource directory).
    pub fn base_dir(&self) -> Option<&Path> {
        self.base_dir.as_deref()
    }

    /// Read the content of a resource file (decoded as UTF-8 text).
    ///
    /// `name` is a relative resource path (e.g. `references/style.md`);
    /// absolute paths and paths containing `..` are rejected. **Symlink
    /// defense**: after `canonicalize`, the target must still be inside
    /// the skill root; links pointing outside the directory (which
    /// malicious skill packages could use to read arbitrary files) return
    /// [`SkillError::NotFound`].
    ///
    /// # Errors
    ///
    /// - the skill comes from text parsing and has no resource directory,
    ///   or the resource does not exist / the path is invalid / the link
    ///   escapes the skill directory → [`SkillError::NotFound`];
    /// - filesystem errors while reading → [`SkillError::Io`].
    pub async fn load_reference(&self, name: &str) -> Result<String, SkillError> {
        let Some(base) = &self.base_dir else {
            return Err(SkillError::NotFound(
                "no resource directory: skill parsed from text".into(),
            ));
        };
        let name_path = Path::new(name);
        if name_path.is_absolute()
            || name_path.components().any(|c| {
                matches!(
                    c,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(SkillError::NotFound(format!(
                "invalid resource path: {name}"
            )));
        }
        let base = tokio::fs::canonicalize(base)
            .await
            .map_err(|e| SkillError::Io(format!("cannot resolve skill root: {e}")))?;
        let canonical = tokio::fs::canonicalize(base.join(name_path))
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    SkillError::NotFound(format!("resource not found: {name}"))
                }
                _ => SkillError::Io(e.to_string()),
            })?;
        if !canonical.starts_with(&base) {
            return Err(SkillError::NotFound(format!(
                "resource escapes skill root: {name}"
            )));
        }
        tokio::fs::read_to_string(canonical)
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    SkillError::NotFound(format!("resource not found: {name}"))
                }
                _ => SkillError::Io(e.to_string()),
            })
    }
}

/// Skill registry: holds a collection of skills, responsible for lookup
/// and disclosure by name.
///
/// Internally an ordered name → skill map guarded by a read-write lock
/// (read-heavy, O(1) lookup by name, registration order preserved);
/// `add` / `remove` take `&self`, so the application can hold the registry
/// handle and **hot-swap** — add/remove does not depend on
/// construction-time ordering, and the next read (menu / lookup) takes
/// effect immediately.
///
/// Cloning is a deep copy: each registry holds its own independent skill
/// collection, so add/remove do not affect each other.
///
/// # Panics
///
/// If the internal read-write lock gets poisoned (a method panics while
/// still holding it), subsequent calls panic. Normal operation (no public
/// method enters a panic path) does not trigger this.
///
/// # Example
///
/// ```
/// use molo::skill::{Skill, SkillRegistry};
///
/// let registry = SkillRegistry::new();
/// let skill = Skill::parse("---\nname: greet\ndescription: Say hello\n---\nHello!").unwrap();
/// registry.add(skill);
///
/// assert_eq!(registry.menu(), "- greet: Say hello");
/// assert_eq!(registry.get("greet").unwrap().body(), "Hello!");
/// // re-registering the same name: replaces the original skill, position
/// // unchanged
/// let v2 = Skill::parse("---\nname: greet\ndescription: Say hello\n---\nGood morning!").unwrap();
/// registry.add(v2);
/// assert_eq!(registry.get("greet").unwrap().body(), "Good morning!");
/// assert_eq!(registry.skills().len(), 1);
/// ```
#[derive(Default)]
pub struct SkillRegistry {
    /// Ordered name → skill map: O(1) lookup by name while preserving
    /// registration order (order affects the menu and static assembly
    /// presentation).
    skills: RwLock<IndexMap<String, Skill>>,
}

impl Clone for SkillRegistry {
    /// std RwLock has no Clone: copy the contents under the lock and
    /// rebuild.
    fn clone(&self) -> Self {
        let skills = self
            .skills
            .read()
            .expect("SkillRegistry internal lock poisoned")
            .clone();
        Self {
            skills: RwLock::new(skills),
        }
    }
}

impl SkillRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a skill; a same-named skill **replaces** the original
    /// (position unchanged), returning `self` for chaining.
    ///
    /// Registration is an explicit operation: a skill must first pass
    /// validation via [`Skill::parse`] / [`Skill::from_dir`]; invalid
    /// skills cannot enter the registry.
    pub fn add(&self, skill: Skill) -> &Self {
        let mut guard = self
            .skills
            .write()
            .expect("SkillRegistry internal lock poisoned");
        guard.insert(skill.name.clone(), skill);
        self
    }

    /// Remove a skill (by name); returns `true` when removed, `false`
    /// when the skill does not exist.
    ///
    /// This is the developer's physical management interface (upgrading /
    /// retiring skills); session-level "invisibility" filtering uses the
    /// assembly layer's allowlist ([`ReActAgent::with_enabled_skills`](crate::agent::ReActAgent::with_enabled_skills)),
    /// and metadata can stay in the registry.
    pub fn remove(&self, name: &str) -> bool {
        let mut guard = self
            .skills
            .write()
            .expect("SkillRegistry internal lock poisoned");
        guard.shift_remove(name).is_some()
    }

    /// Get a skill by name (**cloned**, so the lock-held reference does
    /// not escape; O(1) lookup by name); returns `None` when missing.
    pub fn get(&self, name: &str) -> Option<Skill> {
        let guard = self
            .skills
            .read()
            .expect("SkillRegistry internal lock poisoned");
        guard.get(name).cloned()
    }

    /// Scan a directory and discover all skill directories within it
    /// (reading `SKILL.md` from each subdirectory).
    ///
    /// **Lenient discovery**: bad skills (parse failure / directory name
    /// mismatch / no SKILL.md) are skipped with the reason logged via
    /// `tracing::warn`, without taking down the whole registry; only an
    /// unreadable root directory itself returns an error.
    ///
    /// # Errors
    ///
    /// An unreadable root directory → [`SkillError::Io`].
    pub async fn from_dir(path: &Path) -> Result<Self, SkillError> {
        let registry = SkillRegistry::new();
        let mut entries = tokio::fs::read_dir(path).await?;
        let mut dirs = Vec::new();
        // Directory iteration / type probing failures: skip the entry
        // without taking down the whole discovery.
        loop {
            match entries.next_entry().await {
                Ok(Some(entry)) => {
                    let is_dir = match entry.file_type().await {
                        Ok(ft) => ft.is_dir(),
                        Err(_) => false,
                    };
                    if is_dir {
                        dirs.push(entry.path());
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    // Single-entry read failure: warn and skip, do not
                    // interrupt other directories.
                    tracing::warn!("failed to read skill directory entry: {e}");
                    continue;
                }
            }
        }
        for dir in dirs {
            match Skill::from_dir(&dir).await {
                Ok(skill) => {
                    registry.add(skill);
                }
                Err(err) => tracing::warn!("skipping skill directory {}: {err}", dir.display()),
            }
        }
        Ok(registry)
    }
    /// Discover skills from multiple directories and merge (multi-source
    /// loading).
    ///
    /// Typical scenario: user-level and project-level skill directories as
    /// two sources (concrete paths are the caller's decision; this library
    /// does not hardcode directory conventions). Directories are scanned
    /// in argument order, and **later-loaded skills with the same name
    /// override earlier ones** (argument order is priority: put the
    /// project level last so it overrides the user level).
    ///
    /// **Lenient discovery**: missing directories, unreadable directories
    /// and bad skills are all skipped with the reason logged via
    /// `tracing::warn`, without taking down other sources; if all sources
    /// are unusable, an empty registry is returned.
    ///
    /// # Example
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() {
    /// use molo::skill::SkillRegistry;
    ///
    /// // neither source exists: skipped leniently, resulting in an empty
    /// // registry
    /// let skills =
    ///     SkillRegistry::from_dirs(&["molo-nonexistent-a", "molo-nonexistent-b"]).await;
    /// assert!(skills.skills().is_empty());
    /// # }
    /// ```
    pub async fn from_dirs<P: AsRef<Path>>(paths: &[P]) -> Self {
        let registry = SkillRegistry::new();
        for path in paths {
            let path = path.as_ref();
            match Self::from_dir(path).await {
                Ok(found) => {
                    for skill in found.skills() {
                        registry.add(skill);
                    }
                }
                Err(err) => {
                    tracing::warn!("skipping skill source directory {}: {err}", path.display());
                }
            }
        }
        registry
    }

    /// Disclosure block: one line `- {name}: {description}` per skill, in
    /// registration order.
    ///
    /// This is the first step of progressive disclosure — the model uses
    /// it to decide whether to load a skill by name; one line per skill
    /// keeps the resident system-prompt cost fixed and negligible.
    pub fn menu(&self) -> String {
        let guard = self
            .skills
            .read()
            .expect("SkillRegistry internal lock poisoned");
        let mut out = String::new();
        for (i, skill) in guard.values().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&format!("- {}: {}", skill.name, skill.description));
        }
        out
    }

    /// A cloned snapshot of all skills (in registration order).
    ///
    /// For static assembly scenarios (e.g. the assembly layer's
    /// `with_skills_inline`): take the snapshot and build the system
    /// prompt yourself, bypassing the disclosure flow.
    pub fn skills(&self) -> Vec<Skill> {
        let guard = self
            .skills
            .read()
            .expect("SkillRegistry internal lock poisoned");
        guard.values().cloned().collect()
    }
}

impl std::fmt::Debug for SkillRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Do not block when the lock is held (try_read fails): Debug is
        // only for display.
        match self.skills.try_read() {
            Ok(guard) => f
                .debug_list()
                .entries(guard.values().map(|s| s.name.as_str()))
                .finish(),
            Err(_) => f.write_str("<locked>"),
        }
    }
}

/// Skill loading tool: reads the SKILL.md body by name; the second step of
/// progressive disclosure.
///
/// The model picks a skill from the system-prompt menu and calls this tool
/// (argument `name`), which returns the skill body as the tool result —
/// the body then enters the conversation ledger as protected tool output
/// ([`ToolMemoryPolicy::Protected`]), exempt from window trimming, so skill
/// instructions persist in long conversations.
///
/// The returned content is wrapped in structured tags (so the model can
/// distinguish skill instructions from ordinary conversation content, and
/// context compression can recognize it), and lists the resource files
/// (not pre-read; the model reads them on demand via
/// [`Skill::load_reference`]). **In-session deduplication**: re-loading an
/// already activated skill returns a notice without re-injecting the body
/// (the body is already resident; re-injection would be pure waste).
///
/// The body length should stay within 5000 tokens / 500 lines; anything
/// beyond goes into `references/` resources read via
/// [`Skill::load_reference`] (protocol recommendation, not enforced).
///
/// The assembly layer ([`ReActAgent::with_skills`](crate::agent::ReActAgent::with_skills))
/// automatically registers this tool into the ToolRegistry; `enabled` is
/// the session allowlist view (a snapshot built at construction, `None` =
/// all visible), and skills outside the allowlist are refused. The `name`
/// argument is constrained by enum to allowlisted skill names (queried
/// fresh from the registry each turn, so hot-swaps are reflected per
/// turn), preventing the model from hallucinating nonexistent skills.
///
/// # Errors
///
/// - missing `name` argument → [`ToolError::InvalidArguments`];
/// - the skill exists but is not in the allowlist → [`ToolError::Execution`]("skill not enabled");
/// - the skill does not exist → [`ToolError::Execution`]("skill not found").
///
/// Errors are passed back to the model by the agent loop as ToolResult
/// text, and the model decides what to do next.
#[derive(Debug, Clone)]
pub struct LoadSkillTool {
    registry: Arc<SkillRegistry>,
    enabled: Option<Arc<HashSet<String>>>,
    /// Skills activated in this session (dedup): created and owned by this
    /// tool at construction, not dependent on assembly-layer state.
    activated: Arc<RwLock<HashSet<String>>>,
}

impl LoadSkillTool {
    /// Construct: holds a registry handle and an optional allowlist
    /// (`None` = all skills visible).
    pub fn new(registry: Arc<SkillRegistry>, enabled: Option<Arc<HashSet<String>>>) -> Self {
        Self {
            registry,
            enabled,
            activated: Arc::new(RwLock::new(HashSet::new())),
        }
    }
}

#[async_trait::async_trait]
impl Tool for LoadSkillTool {
    fn schema(&self) -> ToolSchema {
        // Base schema generated by schemars from the argument type (same
        // as tools generated by the `#[molo::tool]` macro); the name enum
        // is a runtime constraint — query the registry for allowlisted
        // skill names, updated per turn as hot-swaps happen (empty = no
        // usable skills, so the model will not call).
        let mut parameters = serde_json::to_value(schemars::schema_for!(LoadSkillArgs))
            .expect("LoadSkillArgs JSON Schema serialization must not fail");
        let available: Vec<String> = self
            .registry
            .skills()
            .iter()
            .filter(|s| self.is_enabled(s.name()))
            .map(|s| s.name().to_string())
            .collect();
        parameters["properties"]["name"]["enum"] = serde_json::json!(available);
        ToolSchema::new(
            "load_skill",
            "Load and activate a skill: the name argument is the skill name, and the skill body is returned. The available skills are listed in the system prompt.",
            parameters,
        )
        .with_policy(ToolPolicy {
            memory_policy: ToolMemoryPolicy::Protected,
            ..Default::default()
        })
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        // Argument parsing is the same as for tools generated by
        // `#[molo::tool]`: serde deserialization, failures classified as
        // InvalidArguments (error text passed back to the model).
        let name = serde_json::from_value::<LoadSkillArgs>(arguments)
            .map_err(ToolError::from)?
            .name;
        if !self.is_enabled(&name) {
            return Err(ToolError::Execution(format!(
                "skill '{name}' is not enabled"
            )));
        }
        let skill = match self.registry.get(&name) {
            Some(skill) => skill,
            None => return Err(ToolError::Execution(format!("skill '{name}' not found"))),
        };
        // In-session dedup: the body is protected and resident, so
        // already activated skills are not re-injected (the notice lets
        // the model know it is "already loaded" and not to call again).
        let mut activated = self
            .activated
            .write()
            .expect("LoadSkillTool internal lock poisoned");
        if activated.contains(&name) {
            return Ok(ToolOutput::text(format!(
                "skill '{name}' is already active in this conversation"
            ))
            .with_memory_policy(ToolMemoryPolicy::Protected)
            .into());
        }
        activated.insert(name);
        Ok(ToolOutput::text(format_skill_content(&skill))
            .with_memory_policy(ToolMemoryPolicy::Protected)
            .into())
    }
}

/// Arguments for load_skill (defined the same way as tools generated by
/// `#[molo::tool]`: serde deserialization + schemars-generated JSON
/// Schema).
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct LoadSkillArgs {
    /// Skill name (the available skills are listed in the system-prompt
    /// menu).
    name: String,
}

impl LoadSkillTool {
    /// Whether the skill is in the session allowlist (no allowlist = all
    /// visible).
    fn is_enabled(&self, name: &str) -> bool {
        match &self.enabled {
            None => true,
            Some(enabled) => enabled.contains(name),
        }
    }
}

/// Assemble the load result: structured tags wrapping the body + resource
/// list (not pre-read).
fn format_skill_content(skill: &Skill) -> String {
    let mut out = String::new();
    out.push_str(&format!("<skill_content name=\"{}\">\n", skill.name()));
    out.push_str(skill.body());
    if skill.base_dir().is_some() {
        // Declare only relative semantics, no absolute paths: filesystem
        // layout must not enter the model context; concrete relative
        // paths are resolved tool-side against the skill root.
        out.push_str("\n\nRelative paths in this skill are relative to the skill directory.");
    }
    if !skill.resources().is_empty() {
        out.push_str("\n\n<skill_resources>");
        for resource in skill.resources() {
            out.push_str(&format!("\n  <file>{}</file>", resource.display()));
        }
        out.push_str("\n</skill_resources>");
    }
    out.push_str("\n</skill_content>");
    out
}

/// Reasons a skill parse / load fails.
///
/// `#[non_exhaustive]` ensures future error categories are not breaking
/// changes; external crates should match with a wildcard arm. The error
/// type is well-behaved: the `Io` variant carries stringified error text
/// (io::Error itself is not Clone; stringifying keeps Clone + PartialEq).
///
/// # Example
///
/// ```
/// use molo::skill::SkillError;
///
/// let err = SkillError::InvalidName("name may only contain lowercase letters, digits, and hyphens".into());
/// assert!(err.to_string().contains("name"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SkillError {
    /// frontmatter structural errors: missing delimiters, syntax errors,
    /// unsupported nested structures, an over-long `compatibility`, etc.
    #[error("invalid frontmatter: {0}")]
    InvalidFrontmatter(String),
    /// `name` missing or not conforming to kebab-case rules.
    #[error("invalid skill name: {0}")]
    InvalidName(String),
    /// `description` missing, empty, or over 1024 characters.
    #[error("invalid skill description: {0}")]
    InvalidDescription(String),
    /// During directory loading, the skill name does not match the
    /// directory name.
    #[error("skill name '{name}' does not match directory name '{dir}'")]
    NameMismatch {
        /// The name in the skill frontmatter.
        name: String,
        /// The directory name.
        dir: String,
    },
    /// Target not found: no SKILL.md in the directory, resource missing,
    /// or a text-parsed skill without a resource directory.
    #[error("skill not found: {0}")]
    NotFound(String),
    /// Body too long (limit in `MAX_SKILL_BODY_CHARS`): the body is
    /// recorded as protected resident context and must not grow unbounded.
    #[error("invalid skill body: {0}")]
    InvalidBody(String),
    /// Filesystem error (stringified; the error is already fully expressed
    /// as text).
    #[error("io error: {0}")]
    Io(String),
}

impl From<std::io::Error> for SkillError {
    fn from(err: std::io::Error) -> Self {
        SkillError::Io(err.to_string())
    }
}

/// Validate a skill name: kebab-case (1-64 characters, lowercase letters /
/// digits / hyphens, not starting or ending with a hyphen, no consecutive
/// hyphens).
fn validate_name(name: &str) -> Result<(), SkillError> {
    if name.is_empty() {
        return Err(SkillError::InvalidName("name must not be empty".into()));
    }
    if name.chars().count() > 64 {
        return Err(SkillError::InvalidName("name exceeds 64 characters".into()));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(SkillError::InvalidName(
            "name may only contain lowercase letters, digits, and hyphens".into(),
        ));
    }
    if name.starts_with('-') || name.ends_with('-') || name.contains("--") {
        return Err(SkillError::InvalidName(
            "name is not kebab-case: must not start or end with a hyphen, and must not contain consecutive hyphens".into(),
        ));
    }
    Ok(())
}

/// Validate a skill description: non-empty, at most 1024 characters.
fn validate_description(description: &str) -> Result<(), SkillError> {
    if description.is_empty() {
        return Err(SkillError::InvalidDescription(
            "description must not be empty".into(),
        ));
    }
    if description.chars().count() > 1024 {
        return Err(SkillError::InvalidDescription(
            "description exceeds 1024 characters".into(),
        ));
    }
    Ok(())
}

/// frontmatter parsing result (raw fields; validation is deferred to
/// `Skill::parse`).
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    license: Option<String>,
    compatibility: Option<String>,
    metadata: Vec<(String, String)>,
    allowed_tools: Vec<AllowedTool>,
}

/// The block context while parsing lines: decides where indented lines
/// belong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Block {
    /// Top level (no block).
    None,
    /// Inside a `metadata:` block: indented `key: value` lines belong to
    /// metadata.
    Metadata,
    /// Inside an `allowed-tools:` block: indented `- item` lines belong to
    /// allowed-tools.
    AllowedTools,
}

/// A hand-written minimal frontmatter parser (zero dependencies)
/// supporting a protocol field subset and both string / list
/// `allowed-tools` forms; unknown top-level scalar fields are tolerated
/// and merged into metadata; nested structures error out.
fn parse_frontmatter(content: &str) -> Result<(Frontmatter, String), SkillError> {
    // Lenient prefix: allow a BOM and leading blank lines (common in real
    // skill files).
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let content = content.trim_start_matches(['\n', '\r']);

    let rest = content.strip_prefix("---").ok_or_else(|| {
        SkillError::InvalidFrontmatter("missing frontmatter start delimiter ---".into())
    })?;
    let (first_line, mut rest) = match rest.split_once('\n') {
        Some((line, tail)) => (line, tail),
        None => (rest, ""),
    };
    if !first_line.trim().is_empty() {
        return Err(SkillError::InvalidFrontmatter(
            "start delimiter --- must be followed by a newline".into(),
        ));
    }

    let mut fm = Frontmatter {
        name: None,
        description: None,
        license: None,
        compatibility: None,
        metadata: Vec::new(),
        allowed_tools: Vec::new(),
    };
    let mut block = Block::None;

    loop {
        let (line, tail) = match rest.split_once('\n') {
            Some((line, tail)) => (line, tail),
            None => (rest, ""),
        };
        if line.trim_end().trim() == "---" {
            // End delimiter: everything after it is the body (a single
            // blank line right after the delimiter is stripped).
            let body = tail.strip_prefix('\n').unwrap_or(tail);
            return Ok((fm, body.to_string()));
        }
        if tail.is_empty() {
            // Reached end of file without seeing the end delimiter.
            return Err(SkillError::InvalidFrontmatter(
                "missing frontmatter end delimiter ---".into(),
            ));
        }

        let trimmed = line.trim_end_matches('\r').trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            // Blank lines and comments: do not change the current block
            // context.
        } else if line.starts_with(' ') || line.starts_with('\t') {
            // Indented line: belongs to the current block.
            if let Some(item) = trimmed.strip_prefix("- ") {
                let item = item.trim();
                if block == Block::Metadata {
                    return Err(SkillError::InvalidFrontmatter(
                        "metadata does not support nested lists".into(),
                    ));
                }
                fm.allowed_tools.push(parse_allowed_tool(item)?);
            } else if block == Block::Metadata {
                let (key, value) = split_kv(trimmed)?;
                fm.metadata
                    .push((key.to_string(), stringify(value.unwrap_or_default())));
            } else {
                return Err(SkillError::InvalidFrontmatter(format!(
                    "unsupported nested structure: {trimmed}"
                )));
            }
        } else {
            // Top-level line: end the current block.
            block = Block::None;
            let (key, value) = split_kv(trimmed)?;
            // Values are uniformly stringified (quotes stripped) before
            // dispatch.
            let value = value.map(stringify);
            match key {
                "name" => fm.name = Some(value.unwrap_or_default().to_string()),
                "description" => fm.description = Some(value.unwrap_or_default().to_string()),
                "license" => fm.license = Some(value.unwrap_or_default().to_string()),
                "compatibility" => {
                    let v = value.unwrap_or_default().to_string();
                    if v.chars().count() > 500 {
                        return Err(SkillError::InvalidFrontmatter(
                            "compatibility exceeds 500 characters".into(),
                        ));
                    }
                    fm.compatibility = Some(v);
                }
                "allowed-tools" => {
                    block = Block::AllowedTools;
                    if let Some(v) = value {
                        // String form: space-separated entries; flow list
                        // form (`[Bash, Python]`) is split on commas.
                        let v = v.trim();
                        if v.starts_with('[') {
                            let inner = v
                                .strip_prefix('[')
                                .and_then(|s| s.strip_suffix(']'))
                                .ok_or_else(|| {
                                    SkillError::InvalidFrontmatter(format!(
                                        "allowed-tools flow list has unbalanced brackets: {v}"
                                    ))
                                })?;
                            for item in inner.split(',') {
                                fm.allowed_tools.push(parse_allowed_tool(item.trim())?);
                            }
                        } else {
                            for item in v.split_whitespace() {
                                fm.allowed_tools.push(parse_allowed_tool(item)?);
                            }
                        }
                    }
                }
                "metadata" => {
                    block = Block::Metadata;
                    if value.is_some() {
                        return Err(SkillError::InvalidFrontmatter(
                            "metadata value must be a key-value block (inline form is not supported)".into(),
                        ));
                    }
                }
                _ => {
                    // Unknown top-level fields are tolerated: merged into
                    // metadata (values already stringified).
                    fm.metadata
                        .push((key.to_string(), value.unwrap_or_default()));
                }
            }
        }
        rest = tail;
    }
}

/// Split a `key: value` line; the key must be non-empty and contain only
/// alphanumerics, hyphens and underscores; the value may be empty.
fn split_kv(line: &str) -> Result<(&str, Option<&str>), SkillError> {
    let Some((key, value)) = line.split_once(':') else {
        return Err(SkillError::InvalidFrontmatter(format!(
            "frontmatter line missing colon: {line}"
        )));
    };
    let key = key.trim();
    if key.is_empty() {
        return Err(SkillError::InvalidFrontmatter(
            "frontmatter line missing field name".into(),
        ));
    }
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(SkillError::InvalidFrontmatter(format!(
            "invalid field name: {key}"
        )));
    }
    let value = value.trim();
    Ok((key, if value.is_empty() { None } else { Some(value) }))
}

/// Stringify a value: strip matching surrounding quotes; everything else
/// (numbers / booleans / text) is kept as-is.
fn stringify(value: &str) -> String {
    let value = value.trim();
    let stripped = if (value.starts_with('"') && value.ends_with('"') && value.len() >= 2)
        || (value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2)
    {
        &value[1..value.len() - 1]
    } else {
        value
    };
    stripped.to_string()
}

/// Parse an allowed-tools entry: `name(scope)` or `name`.
fn parse_allowed_tool(item: &str) -> Result<AllowedTool, SkillError> {
    if let Some(open) = item.find('(') {
        if !item.ends_with(')') || item[open + 1..].contains('(') {
            return Err(SkillError::InvalidFrontmatter(format!(
                "invalid allowed-tools entry: {item}"
            )));
        }
        let name = item[..open].trim();
        if name.is_empty() {
            return Err(SkillError::InvalidFrontmatter(format!(
                "allowed-tools entry missing tool name: {item}"
            )));
        }
        let scope = item[open + 1..item.len() - 1].trim();
        Ok(AllowedTool {
            name: name.to_string(),
            scope: (!scope.is_empty()).then(|| scope.to_string()),
        })
    } else if item.contains(')') {
        // Closing parenthesis without an opening one: malformed entry
        // (e.g. Bash)git:*).
        Err(SkillError::InvalidFrontmatter(format!(
            "invalid allowed-tools entry: {item}"
        )))
    } else if item.contains(['[', ']', ',']) {
        // Residual list syntax (e.g. "[Bash," / "Python]" split out by
        // spaces): error out explicitly instead of silently accepting it
        // as a tool name.
        Err(SkillError::InvalidFrontmatter(format!(
            "invalid allowed-tools entry: {item}"
        )))
    } else if item.is_empty() {
        Err(SkillError::InvalidFrontmatter(
            "empty allowed-tools entry".into(),
        ))
    } else {
        Ok(AllowedTool {
            name: item.to_string(),
            scope: None,
        })
    }
}

/// Recursively collect all files under the resource directories
/// (references / scripts / assets), returned as paths relative to the
/// skill root, sorted for determinism.
async fn collect_resources(base: &Path) -> Vec<PathBuf> {
    let mut resources = Vec::new();
    for dir in ["references", "scripts", "assets"] {
        walk_dir(base.join(dir), base, &mut resources).await;
    }
    resources.sort();
    resources
}

async fn walk_dir(dir: PathBuf, base: &Path, out: &mut Vec<PathBuf>) {
    let mut entries = match tokio::fs::read_dir(&dir).await {
        Ok(e) => e,
        Err(_) => return, // directory missing / unreadable: ignore
    };
    loop {
        match entries.next_entry().await {
            Ok(Some(entry)) => {
                let path = entry.path();
                let is_dir = match entry.file_type().await {
                    Ok(ft) => ft.is_dir(),
                    Err(_) => false,
                };
                if is_dir {
                    Box::pin(walk_dir(path, base, out)).await;
                } else if let Ok(rel) = path.strip_prefix(base) {
                    out.push(rel.to_path_buf());
                }
            }
            Ok(None) => break,
            Err(_) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    // Skill component tests: parse validation (incl. unknown-field
    // tolerance) / directory discovery (incl. bad-skill skipping) / both
    // allowed-tools forms / same-name replacement / hot-swap add-remove /
    // menu format / resource reading.

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use super::{AllowedTool, LoadSkillTool, Skill, SkillError, SkillRegistry};
    use crate::tool::Tool;
    use std::collections::HashSet;

    /// Create a temp directory (unique per process id + tag, so tests do
    /// not conflict).
    /// Test temp directory: cleaned up automatically on drop, leaving no
    /// residue in /tmp.
    fn temp_dir(tag: &str) -> TempDir {
        TempDir::new(tag)
    }

    /// Test temp directory handle: `Deref`s to `PathBuf`, deletes the
    /// whole directory on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("molo-skill-test-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl std::ops::Deref for TempDir {
        type Target = PathBuf;
        fn deref(&self) -> &PathBuf {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Write a skill directory (SKILL.md plus optional resource files),
    /// returning the directory path.
    fn write_skill(dir: &Path, name: &str, description: &str, body: &str) -> PathBuf {
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n{body}"),
        )
        .unwrap();
        skill_dir
    }

    fn minimal(name: &str) -> Skill {
        Skill::parse(&format!(
            "---\nname: {name}\ndescription: description\n---\nbody"
        ))
        .unwrap()
    }

    // ---------- parsing: valid input ----------

    #[test]
    fn parse_minimal() {
        let skill = Skill::parse("---\nname: greet\ndescription: Say hello\n---\nHello!").unwrap();
        assert_eq!(skill.name(), "greet");
        assert_eq!(skill.description(), "Say hello");
        assert_eq!(skill.body(), "Hello!");
        assert!(skill.license().is_none());
        assert!(skill.metadata().is_empty());
        assert!(skill.allowed_tools().is_empty());
        assert!(skill.resources().is_empty());
    }

    #[test]
    fn parse_full_fields() {
        let content = r#"---
name: code-review
description: Review code changes
license: MIT
compatibility: rust-1.80+
metadata:
  author: team
  public: true
allowed-tools:
  - Bash(git:*)
  - Python
user-invocable: true
---
Review steps.
"#;
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name(), "code-review");
        assert_eq!(skill.license(), Some("MIT"));
        assert_eq!(skill.compatibility(), Some("rust-1.80+"));
        // metadata: explicit block + unknown top-level fields
        // (stringified).
        assert_eq!(
            skill.metadata(),
            &[
                ("author".to_string(), "team".to_string()),
                ("public".to_string(), "true".to_string()),
                ("user-invocable".to_string(), "true".to_string()),
            ]
        );
        // allowed-tools: list form (scoped + unscoped).
        assert_eq!(
            skill.allowed_tools(),
            &[
                AllowedTool {
                    name: "Bash".into(),
                    scope: Some("git:*".into())
                },
                AllowedTool {
                    name: "Python".into(),
                    scope: None
                },
            ]
        );
    }

    #[test]
    fn parse_allowed_tools_string_form() {
        let skill = Skill::parse(
            "---\nname: a\ndescription: description\nallowed-tools: Bash(git:*) Python\n---\nbody",
        )
        .unwrap();
        assert_eq!(skill.allowed_tools().len(), 2);
        assert_eq!(skill.allowed_tools()[0].name, "Bash");
        assert_eq!(skill.allowed_tools()[0].scope.as_deref(), Some("git:*"));
        assert_eq!(skill.allowed_tools()[1].name, "Python");
    }

    #[test]
    fn parse_allowed_tools_flow_list_form() {
        // flow style `[Bash, Python]`: parsed on commas.
        let skill = Skill::parse(
            "---\nname: a\ndescription: description\nallowed-tools: [Bash, Python]\n---\nbody",
        )
        .unwrap();
        assert_eq!(skill.allowed_tools().len(), 2);
        assert_eq!(skill.allowed_tools()[0].name, "Bash");
        assert_eq!(skill.allowed_tools()[1].name, "Python");
    }

    #[test]
    fn parse_allowed_tools_flow_list_unbalanced_brackets_rejected() {
        // Unbalanced brackets: explicit error instead of silently
        // producing garbage tool names.
        let err =
            Skill::parse("---\nname: a\ndescription: description\nallowed-tools: [Bash\n---\nbody")
                .unwrap_err();
        assert!(err.to_string().contains("unbalanced brackets"));
    }

    #[test]
    fn frontmatter_underscore_keys_tolerated_into_metadata() {
        // Underscore keys (e.g. user_invocable) do not error; unknown keys
        // are tolerated and merged into metadata.
        let skill =
            Skill::parse("---\nname: a\ndescription: description\nuser_invocable: true\n---\nbody")
                .unwrap();
        assert_eq!(
            skill.metadata(),
            &[("user_invocable".to_string(), "true".to_string())]
        );
    }

    #[test]
    fn parse_body_with_blank_line_after_delimiter() {
        // A blank line immediately after the end delimiter is formatting;
        // strip it.
        let skill = Skill::parse(
            "---\nname: a\ndescription: description\n---\n\nfirst body line\n\nsecond body line",
        )
        .unwrap();
        assert_eq!(skill.body(), "first body line\n\nsecond body line");
    }

    #[test]
    fn parse_empty_body_allowed() {
        // The body may be empty: a skill with only frontmatter.
        let skill = Skill::parse("---\nname: a\ndescription: description\n---").unwrap();
        assert_eq!(skill.body(), "");
    }

    #[test]
    fn parse_bom_and_leading_blank_lines_tolerated() {
        let content = "\u{feff}\n\n---\nname: a\ndescription: description\n---\nbody";
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name(), "a");
    }

    #[test]
    fn parse_quoted_values_stripped() {
        let skill =
            Skill::parse("---\nname: a\ndescription: \"quoted description\"\n---\n").unwrap();
        assert_eq!(skill.description(), "quoted description");
    }

    #[test]
    fn parse_comments_and_blank_lines_ignored() {
        let content = "---\n# this is a comment\n\nname: a\ndescription: description\n---\nbody";
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name(), "a");
    }

    #[test]
    fn parse_duplicate_field_last_wins() {
        let content = "---\nname: a\ndescription: first\ndescription: second\n---\n";
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.description(), "second");
    }

    // ---------- parsing: invalid input (explicit operations error
    // strictly) ----------

    #[test]
    fn parse_missing_frontmatter() {
        let err = Skill::parse("plain text, no frontmatter").unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter(_)));
    }

    #[test]
    fn parse_missing_end_delimiter() {
        let err =
            Skill::parse("---\nname: a\ndescription: description\nbody without end delimiter")
                .unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter(_)));
    }

    #[test]
    fn parse_missing_name() {
        let err = Skill::parse("---\ndescription: description\n---\n").unwrap_err();
        assert!(matches!(err, SkillError::InvalidName(_)));
    }

    #[test]
    fn parse_invalid_names() {
        // uppercase letters / illegal characters / too long / leading or
        // trailing hyphens / consecutive hyphens.
        for bad in [
            "Bad-name",
            "bad_name",
            "bad name",
            &"a".repeat(65),
            "-bad",
            "bad-",
            "ba--d",
        ] {
            let content = format!("---\nname: {bad}\ndescription: description\n---\n");
            assert!(
                matches!(Skill::parse(&content), Err(SkillError::InvalidName(_))),
                "name should be rejected: {bad}"
            );
        }
    }

    #[test]
    fn parse_invalid_descriptions() {
        let missing = Skill::parse("---\nname: a\n---\n").unwrap_err();
        assert!(matches!(missing, SkillError::InvalidDescription(_)));

        let long = Skill::parse(&format!(
            "---\nname: a\ndescription: {}\n---\n",
            "x".repeat(1025)
        ))
        .unwrap_err();
        assert!(matches!(long, SkillError::InvalidDescription(_)));
    }

    #[test]
    fn parse_compatibility_too_long() {
        let err = Skill::parse(&format!(
            "---\nname: a\ndescription: description\ncompatibility: {}\n---\n",
            "x".repeat(501)
        ))
        .unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter(_)));
    }

    #[test]
    fn parse_metadata_nested_rejected() {
        // Nested list inside metadata: unsupported, error out.
        let content = "---\nname: a\ndescription: description\nmetadata:\n  tags:\n    - x\n---\n";
        let err = Skill::parse(content).unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter(_)));
    }

    #[test]
    fn parse_metadata_inline_value_rejected() {
        let err = Skill::parse("---\nname: a\ndescription: description\nmetadata: foo\n---\n")
            .unwrap_err();
        assert!(matches!(err, SkillError::InvalidFrontmatter(_)));
    }

    #[test]
    fn parse_unknown_field_empty_value() {
        // Unknown top-level field with empty value: tolerated, merged into
        // metadata as an empty string.
        let skill =
            Skill::parse("---\nname: a\ndescription: description\nuser-invocable:\n---\n").unwrap();
        assert_eq!(
            skill.metadata(),
            &[("user-invocable".to_string(), String::new())]
        );
    }

    #[test]
    fn parse_invalid_allowed_tool_entries() {
        // Unclosed parens / trailing content after parens / empty entries.
        for bad in ["Bash(git:*", "Bash)git:*", "()", "(x)"] {
            let content =
                format!("---\nname: a\ndescription: description\nallowed-tools: {bad}\n---\n");
            assert!(
                matches!(
                    Skill::parse(&content),
                    Err(SkillError::InvalidFrontmatter(_))
                ),
                "entry should be rejected: {bad}"
            );
        }
    }

    // ---------- directory loading ----------

    #[tokio::test]
    async fn from_dir_ok_with_resources() {
        let dir = temp_dir("from-dir-ok");
        let skill_dir = write_skill(&dir, "code-review", "Review code", "Step one");
        // Resources: a nested references file + a scripts file + one
        // unrelated file (not collected).
        std::fs::create_dir_all(skill_dir.join("references/nested")).unwrap();
        std::fs::write(skill_dir.join("references/style.md"), "# style").unwrap();
        std::fs::write(skill_dir.join("references/nested/check.md"), "# checklist").unwrap();
        std::fs::create_dir_all(skill_dir.join("scripts")).unwrap();
        std::fs::write(skill_dir.join("scripts/run.sh"), "#!/bin/sh").unwrap();
        std::fs::write(skill_dir.join("README.md"), "not a resource").unwrap();

        let skill = Skill::from_dir(&skill_dir).await.unwrap();
        assert_eq!(skill.name(), "code-review");
        assert_eq!(skill.body(), "Step one");
        // Resources: three files, relative paths, stable ordering.
        assert_eq!(
            skill.resources(),
            &[
                PathBuf::from("references/nested/check.md"),
                PathBuf::from("references/style.md"),
                PathBuf::from("scripts/run.sh"),
            ]
        );
    }

    #[tokio::test]
    async fn from_dir_name_mismatch() {
        let dir = temp_dir("from-dir-mismatch");
        // Directory named wrong-dir, skill named right-name → NameMismatch.
        let skill_dir = dir.join("wrong-dir");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: right-name\ndescription: description\n---\nbody",
        )
        .unwrap();

        let err = Skill::from_dir(&skill_dir).await.unwrap_err();
        assert!(matches!(
            err,
            SkillError::NameMismatch { name, dir: _ } if name == "right-name"
        ));
    }

    #[tokio::test]
    async fn from_dir_missing_skill_md() {
        let dir = temp_dir("from-dir-missing");
        let empty = dir.join("empty-skill");
        std::fs::create_dir_all(&empty).unwrap();

        let err = Skill::from_dir(&empty).await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
    }

    #[tokio::test]
    async fn load_reference_ok() {
        let dir = temp_dir("load-ref");
        let skill_dir = write_skill(&dir, "a", "description", "body");
        std::fs::create_dir_all(skill_dir.join("references")).unwrap();
        std::fs::write(skill_dir.join("references/style.md"), "style content").unwrap();
        let skill = Skill::from_dir(&skill_dir).await.unwrap();

        assert_eq!(
            skill.load_reference("references/style.md").await.unwrap(),
            "style content"
        );
    }

    #[tokio::test]
    async fn load_reference_missing_or_invalid() {
        let dir = temp_dir("load-ref-missing");
        let skill_dir = write_skill(&dir, "a", "description", "body");
        let skill = Skill::from_dir(&skill_dir).await.unwrap();

        // Missing.
        let err = skill.load_reference("nope.md").await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
        // Path traversal rejected.
        let err = skill.load_reference("../SKILL.md").await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
        // Absolute path rejected.
        let err = skill.load_reference("/etc/passwd").await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));

        // Text-parsed skill has no resource directory.
        let parsed = Skill::parse("---\nname: a\ndescription: description\n---\nbody").unwrap();
        let err = parsed.load_reference("x.md").await.unwrap_err();
        assert!(matches!(err, SkillError::NotFound(_)));
    }

    /// Symlink defense: links pointing outside the skill directory must be
    /// rejected (arbitrary file read surface).
    #[cfg(unix)]
    #[tokio::test]
    async fn load_reference_rejects_symlink_escape() {
        let dir = temp_dir("load-ref-symlink");
        let skill_dir = write_skill(&dir, "a", "description", "body");
        std::fs::create_dir_all(skill_dir.join("references")).unwrap();
        // A secret file outside the directory + a symlink pointing at it.
        let secret = dir.join("secret.txt");
        std::fs::write(&secret, "secret content").unwrap();
        std::os::unix::fs::symlink(&secret, skill_dir.join("references/leak")).unwrap();
        let skill = Skill::from_dir(&skill_dir).await.unwrap();

        let err = skill.load_reference("references/leak").await.unwrap_err();
        assert!(
            matches!(err, SkillError::NotFound(_)),
            "symlink escape must be rejected, got: {err:?}"
        );
    }

    /// Symlink defense: links inside the directory (pointing at
    /// in-directory resources) still work.
    #[cfg(unix)]
    #[tokio::test]
    async fn load_reference_allows_internal_symlink() {
        let dir = temp_dir("load-ref-symlink-in");
        let skill_dir = write_skill(&dir, "a", "description", "body");
        std::fs::create_dir_all(skill_dir.join("references")).unwrap();
        std::fs::write(skill_dir.join("references/real.md"), "real content").unwrap();
        std::os::unix::fs::symlink("real.md", skill_dir.join("references/alias.md")).unwrap();
        let skill = Skill::from_dir(&skill_dir).await.unwrap();

        assert_eq!(
            skill.load_reference("references/alias.md").await.unwrap(),
            "real content"
        );
    }

    /// Symlink defense: from_dir rejects when SKILL.md itself is a link to
    /// a file outside the directory.
    #[cfg(unix)]
    #[tokio::test]
    async fn from_dir_rejects_symlinked_skill_md() {
        let dir = temp_dir("from-dir-symlink");
        let skill_dir = dir.join("a");
        std::fs::create_dir_all(&skill_dir).unwrap();
        let secret = dir.join("secret.md");
        std::fs::write(
            &secret,
            "---\nname: a\ndescription: description\n---\nsecret body",
        )
        .unwrap();
        std::os::unix::fs::symlink(&secret, skill_dir.join("SKILL.md")).unwrap();

        let err = Skill::from_dir(&skill_dir).await.unwrap_err();
        assert!(
            matches!(err, SkillError::NotFound(_)),
            "SKILL.md symlink escape must be rejected, got: {err:?}"
        );
    }

    // ---------- registry ----------

    #[test]
    fn registry_add_get_remove() {
        let registry = SkillRegistry::new();
        assert!(registry.get("a").is_none());

        registry.add(minimal("a"));
        assert_eq!(registry.get("a").unwrap().name(), "a");
        assert!(registry.remove("a"));
        assert!(registry.get("a").is_none());
        // Removing again: false.
        assert!(!registry.remove("a"));
    }

    #[test]
    fn registry_add_duplicate_replaces_in_place() {
        let registry = SkillRegistry::new();
        registry
            .add(minimal("a"))
            .add(minimal("b"))
            .add(minimal("c"));

        // Replace b with a same-named skill: position unchanged (order a,
        // b', c).
        let v2 = Skill::parse("---\nname: b\ndescription: new description\n---\nnew body").unwrap();
        registry.add(v2);
        let names: Vec<String> = registry
            .skills()
            .iter()
            .map(|s| s.name().to_string())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        assert_eq!(registry.get("b").unwrap().body(), "new body");
    }

    #[test]
    fn registry_menu_format() {
        let registry = SkillRegistry::new();
        registry.add(minimal("a")).add(minimal("b"));
        assert_eq!(registry.menu(), "- a: description\n- b: description");
        // Empty registry: empty disclosure block.
        assert_eq!(SkillRegistry::new().menu(), "");
    }

    #[tokio::test]
    async fn registry_from_dir_skips_bad_skills() {
        let dir = temp_dir("registry-from-dir");
        write_skill(&dir, "good-one", "good skill", "body");
        // Bad skill 1: directory name does not match the skill name.
        let bad = dir.join("bad-one");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(
            bad.join("SKILL.md"),
            "---\nname: other-name\ndescription: description\n---\nbody",
        )
        .unwrap();
        // Bad skill 2: no SKILL.md.
        std::fs::create_dir_all(dir.join("empty-dir")).unwrap();
        // Unrelated file (not a directory): skipped.
        std::fs::write(dir.join("notes.md"), "not a skill").unwrap();

        let registry = SkillRegistry::from_dir(&dir).await.unwrap();
        let names: Vec<String> = registry
            .skills()
            .iter()
            .map(|s| s.name().to_string())
            .collect();
        assert_eq!(names, vec!["good-one"]);
        assert!(registry.get("bad-one").is_none());
    }

    #[tokio::test]
    async fn from_dirs_merges_with_later_override() {
        let dir = temp_dir("from-dirs-merge");
        let user = dir.join("user");
        let project = dir.join("project");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        // User level: two skills.
        write_skill(&user, "greet", "user version", "user body");
        write_skill(&user, "user-only", "user only", "body");
        // Project level: same-named greet (new body) + a new skill — the
        // project level comes later in the arguments and overrides the
        // user level.
        write_skill(&project, "greet", "project version", "project body");
        write_skill(&project, "project-only", "project only", "body");

        let registry = SkillRegistry::from_dirs(&[user, project]).await;
        let names: Vec<String> = registry
            .skills()
            .iter()
            .map(|s| s.name().to_string())
            .collect();
        // Same-name replacement keeps the first registration position:
        // greet first, body is the project version.
        assert_eq!(names, vec!["greet", "user-only", "project-only"]);
        assert_eq!(registry.get("greet").unwrap().body(), "project body");
        assert_eq!(registry.get("user-only").unwrap().body(), "body");
    }

    #[tokio::test]
    async fn from_dirs_skips_missing_sources() {
        let dir = temp_dir("from-dirs-missing");
        let exists = dir.join("exists");
        std::fs::create_dir_all(&exists).unwrap();
        write_skill(&exists, "a", "description", "body");

        // A missing directory in the middle: skipped, other sources
        // unaffected.
        let registry =
            SkillRegistry::from_dirs(&[dir.join("missing-a"), exists, dir.join("missing-b")]).await;
        assert_eq!(registry.skills().len(), 1);

        // All sources missing: empty registry (lenient, no error).
        let empty = SkillRegistry::from_dirs(&[dir.join("missing-a"), dir.join("missing-b")]).await;
        assert!(empty.skills().is_empty());
    }

    #[tokio::test]
    async fn from_dirs_empty_list() {
        let registry = SkillRegistry::from_dirs::<&str>(&[]).await;
        assert!(registry.skills().is_empty());
    }

    #[tokio::test]
    async fn registry_from_dir_root_io_error() {
        let missing = temp_dir("registry-root").join("does-not-exist");
        let err = SkillRegistry::from_dir(&missing).await.unwrap_err();
        assert!(matches!(err, SkillError::Io(_)));
    }

    #[test]
    fn registry_hot_swap_add_remove() {
        // Hot-swap basics: add / remove take &self; a shared handle (Arc)
        // can read/write concurrently.
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        let handle = Arc::clone(&registry);

        handle.add(minimal("a"));
        assert!(registry.get("a").is_some());
        handle.remove("a");
        assert!(registry.get("a").is_none());
    }

    #[test]
    fn allowed_tool_permits_rules() {
        let bash_git = AllowedTool {
            name: "Bash".into(),
            scope: Some("git:*".into()),
        };
        // Exact name + scope prefix (trailing wildcard stripped).
        assert!(bash_git.permits("Bash", "git:diff --stat"));
        assert!(bash_git.permits("Bash", "git:log"));
        assert!(!bash_git.permits("Bash", "rm -rf /"));
        assert!(!bash_git.permits("Python", "git:log"));

        // Scope without wildcard: pure prefix.
        let exact = AllowedTool {
            name: "Bash".into(),
            scope: Some("git:status".into()),
        };
        assert!(exact.permits("Bash", "git:status"));
        assert!(!exact.permits("Bash", "git:log"));

        // No scope: any arguments for the tool.
        let python = AllowedTool {
            name: "Python".into(),
            scope: None,
        };
        assert!(python.permits("Python", "print('hello')"));
        assert!(!python.permits("Bash", "echo hi"));
    }

    // ---------- LoadSkillTool ----------

    async fn call_load_skill(
        tool: &LoadSkillTool,
        arguments: serde_json::Value,
        state: &crate::SharedState,
    ) -> Result<String, crate::tool::ToolError> {
        let run = crate::RunContext::new("load-skill-test");
        let result = tool
            .call(
                arguments,
                crate::ToolContext::new(&run, state, "call-load-skill", "load_skill"),
            )
            .await?;
        Ok(result.to_string())
    }

    #[tokio::test]
    async fn load_skill_returns_wrapped_body() {
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        registry.add(minimal("a"));
        let tool = LoadSkillTool::new(Arc::clone(&registry), None);
        let state = crate::SharedState::new();

        let result = call_load_skill(&tool, serde_json::json!({ "name": "a" }), &state)
            .await
            .unwrap();
        // Structured tags wrapping + body (text-parsed skill has no
        // resource list).
        assert_eq!(result, "<skill_content name=\"a\">\nbody\n</skill_content>");
    }

    #[tokio::test]
    async fn load_skill_deduplicates_activations() {
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        registry.add(minimal("a"));
        let tool = LoadSkillTool::new(Arc::clone(&registry), None);
        let state = crate::SharedState::new();

        // First call: returns the body and records it in the session
        // activation set.
        let first = call_load_skill(&tool, serde_json::json!({ "name": "a" }), &state)
            .await
            .unwrap();
        assert!(first.contains("body"));
        // Second call: already in context, returns the notice without
        // re-injecting the body.
        let second = call_load_skill(&tool, serde_json::json!({ "name": "a" }), &state)
            .await
            .unwrap();
        assert!(second.contains("already active"));
        assert!(!second.contains("body"));
        // A new instance (new session) has an independent set: can reload.
        let fresh = LoadSkillTool::new(Arc::clone(&registry), None);
        let again = call_load_skill(&fresh, serde_json::json!({ "name": "a" }), &state)
            .await
            .unwrap();
        assert!(again.contains("body"));
    }

    #[tokio::test]
    async fn load_skill_schema_enum_lists_enabled_skills() {
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        registry
            .add(minimal("a"))
            .add(minimal("b"))
            .add(minimal("c"));
        let enabled: Arc<HashSet<String>> =
            Arc::new(["a".to_string(), "b".to_string()].into_iter().collect());
        let tool = LoadSkillTool::new(registry, Some(enabled));

        let schema = tool.schema();
        let names = schema.parameters["properties"]["name"]["enum"]
            .as_array()
            .expect("name should be an enum")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        // Allowlist filter (c excluded); updates after hot-swap are
        // guaranteed by fresh lookups.
        assert_eq!(names, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn load_skill_content_lists_resources() {
        let dir = temp_dir("load-skill-resources");
        let skill_dir = write_skill(&dir, "a", "description", "body");
        std::fs::create_dir_all(skill_dir.join("references")).unwrap();
        std::fs::write(skill_dir.join("references/style.md"), "# style").unwrap();
        let skill = Skill::from_dir(&skill_dir).await.unwrap();

        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        registry.add(skill);
        let tool = LoadSkillTool::new(registry, None);
        let state = crate::SharedState::new();

        let result = call_load_skill(&tool, serde_json::json!({ "name": "a" }), &state)
            .await
            .unwrap();
        assert!(result.contains("<skill_content name=\"a\">"));
        // Only relative semantics are declared, no absolute paths
        // (filesystem layout does not enter the model context).
        assert!(
            result.contains("Relative paths in this skill are relative to the skill directory.")
        );
        assert!(!result.contains(&skill_dir.display().to_string()));
        assert!(result.contains("<skill_resources>"));
        assert!(result.contains("<file>references/style.md</file>"));
        assert!(result.ends_with("</skill_content>"));
    }

    #[tokio::test]
    async fn load_skill_not_found() {
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        let tool = LoadSkillTool::new(registry, None);
        let state = crate::SharedState::new();

        let err = call_load_skill(&tool, serde_json::json!({ "name": "ghost" }), &state)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn load_skill_not_enabled() {
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        registry.add(minimal("a")).add(minimal("b"));
        let enabled: Arc<std::collections::HashSet<String>> =
            Arc::new(["a".to_string()].into_iter().collect());
        let tool = LoadSkillTool::new(registry, Some(enabled));
        let state = crate::SharedState::new();

        // A skill outside the allowlist: not enabled (even though it
        // exists).
        let err = call_load_skill(&tool, serde_json::json!({ "name": "b" }), &state)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not enabled"));
        // Inside the allowlist: normal (tags wrapping the body).
        let ok = call_load_skill(&tool, serde_json::json!({ "name": "a" }), &state)
            .await
            .unwrap();
        assert!(ok.contains("body"));
    }

    #[tokio::test]
    async fn load_skill_missing_name_argument() {
        let registry: Arc<SkillRegistry> = Arc::new(SkillRegistry::new());
        let tool = LoadSkillTool::new(registry, None);
        let state = crate::SharedState::new();
        let err = call_load_skill(&tool, serde_json::json!({}), &state)
            .await
            .unwrap_err();
        assert!(matches!(err, crate::tool::ToolError::InvalidArguments(_)));
    }
}
