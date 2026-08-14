//! A procedural macro that compiles an async function into a `molo::tool::Tool` implementation.
#![warn(missing_docs)]
//!
//! The [`tool`] attribute macro generates all the boilerplate needed for a tool definition in one shot from the function signature: the unit struct,
//! the parameter JSON Schema, and the `call` method (including argument parsing and `&SharedState` injection),
//! so you never have to write `impl Tool` by hand.
//!
//! This crate is usually not used as a direct dependency; it is re-exported by `molo` and invoked as
//! `#[molo::tool(...)]` (`molo::tool` is also the name of a module; attribute macros and
//! modules live in different namespaces, so they can share the same name):
//!
//! ```text
//! #[molo::tool(description = "Evaluate the result of a mathematical expression")]
//! async fn calculator(expression: String) -> Result<String, molo::tool::ToolError> {
//!     // Business logic...
//!     Ok(expression)
//! }
//! ```
//!
//! # Shape constraints
//!
//! - Parameters: 0 or 1 business parameter, plus an optional **trailing** `&SharedState` (an immutable
//!   reference recognized by the type name `SharedState`, injected automatically; must be the last parameter);
//! - The business parameter type must implement `schemars::JsonSchema`: primitive types come with an
//!   implementation out of the box, and custom structs just need `#[derive(JsonSchema)]`;
//! - Return: `Result<String, ToolError>`, `Result<ToolOutput, ToolError>`, or
//!   `Result<ToolResult, ToolError>` (a type mismatch is caught by the compiler on the generated code);
//! - Attributes: `description` (required), `name` (optional, defaults to the function name), `protected`
//!   (optional, defaults to `false`; when `true` the result is protected and exempt from window trimming),
//!   plus optional policy hints: `side_effects`, `risk`, `requires_confirmation`, and `timeout_secs`.
//!
//! # Dependency prerequisites
//!
//! The generated code hard-references the `::molo::`, `::serde_json::`, and `::schemars::` paths:
//! depend on this framework under the crate name `molo` (renaming it would make the generated code unresolvable), and add
//! `serde_json` and `schemars` as direct dependencies — the generated code references them directly, and the
//! parameter struct's `#[derive(JsonSchema)]` also requires `schemars`.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{FnArg, Ident, ItemFn, LitInt, LitStr, Pat, Type, parse_macro_input};

/// Macro attributes: `description` (required) + optional schema policy hints.
struct ToolArgs {
    description: String,
    name: Option<String>,
    /// Whether output is protected: protected results are recorded via
    /// protected memory when the memory implementation supports it.
    protected: bool,
    /// Declared side-effect level.
    side_effects: SideEffectAttr,
    /// Declared default risk.
    risk: RiskAttr,
    /// Whether the tool author recommends confirmation.
    requires_confirmation: bool,
    /// Suggested timeout in seconds.
    timeout_secs: Option<u64>,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut description = None;
        let mut name = None;
        let mut protected = false;
        let mut side_effects = SideEffectAttr::Pure;
        let mut risk = RiskAttr::Low;
        let mut requires_confirmation = false;
        let mut timeout_secs = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            match key.to_string().as_str() {
                "description" => {
                    let value: LitStr = input.parse()?;
                    description = Some(value.value());
                }
                "name" => {
                    let value: LitStr = input.parse()?;
                    name = Some(value.value());
                }
                "protected" => {
                    let value: syn::LitBool = input.parse()?;
                    protected = value.value;
                }
                "side_effects" => {
                    let value: LitStr = input.parse()?;
                    side_effects = SideEffectAttr::parse(value)?;
                }
                "risk" => {
                    let value: LitStr = input.parse()?;
                    risk = RiskAttr::parse(value)?;
                }
                "requires_confirmation" => {
                    let value: syn::LitBool = input.parse()?;
                    requires_confirmation = value.value;
                }
                "timeout_secs" => {
                    let value: LitInt = input.parse()?;
                    timeout_secs = Some(value.base10_parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown attribute {other}: only description / name / protected / side_effects / risk / requires_confirmation / timeout_secs are supported"
                        ),
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<syn::Token![,]>()?;
            }
        }
        Ok(ToolArgs {
            description: description.ok_or_else(|| {
                syn::Error::new(Span::call_site(), "missing required attribute description")
            })?,
            name,
            protected,
            side_effects,
            risk,
            requires_confirmation,
            timeout_secs,
        })
    }
}

#[derive(Clone, Copy)]
enum SideEffectAttr {
    Pure,
    ReadOnly,
    Write,
    External,
}

impl SideEffectAttr {
    fn parse(lit: LitStr) -> syn::Result<Self> {
        match lit.value().as_str() {
            "pure" => Ok(Self::Pure),
            "read_only" => Ok(Self::ReadOnly),
            "write" => Ok(Self::Write),
            "external" => Ok(Self::External),
            other => Err(syn::Error::new(
                lit.span(),
                format!(
                    "invalid side_effects value {other:?}: expected pure / read_only / write / external"
                ),
            )),
        }
    }

    fn tokens(self) -> proc_macro2::TokenStream {
        match self {
            Self::Pure => quote! { ::molo::tool::SideEffectLevel::Pure },
            Self::ReadOnly => quote! { ::molo::tool::SideEffectLevel::ReadOnly },
            Self::Write => quote! { ::molo::tool::SideEffectLevel::Write },
            Self::External => quote! { ::molo::tool::SideEffectLevel::External },
        }
    }
}

#[derive(Clone, Copy)]
enum RiskAttr {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskAttr {
    fn parse(lit: LitStr) -> syn::Result<Self> {
        match lit.value().as_str() {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "critical" => Ok(Self::Critical),
            other => Err(syn::Error::new(
                lit.span(),
                format!("invalid risk value {other:?}: expected low / medium / high / critical"),
            )),
        }
    }

    fn tokens(self) -> proc_macro2::TokenStream {
        match self {
            Self::Low => quote! { ::molo::RiskLevel::Low },
            Self::Medium => quote! { ::molo::RiskLevel::Medium },
            Self::High => quote! { ::molo::RiskLevel::High },
            Self::Critical => quote! { ::molo::RiskLevel::Critical },
        }
    }
}

/// Compiles an async function into a `molo::tool::Tool` implementation.
///
/// At the function definition site, the macro generates a unit struct named after the function
/// (snake_case → PascalCase, e.g. `fetch_weather` → `FetchWeather`) and implements `Tool` for it:
///
/// - `schema()`: a `ToolSchema` derived from the function signature; the tool name defaults to the function name;
/// - `call()`: parses model arguments with the same rules as the schema, calls the original function, and
///   injects the `&SharedState` parameter automatically (ignored when the function does not declare shared state).
///
/// Signature requirements: must be an `async fn` with no generics and no `self`; at most 1 business
/// parameter (define a parameter struct for multiple fields); `&SharedState` must be the **last** parameter, recognized by
/// "an immutable reference + the type name `SharedState` (a bare path or one starting with `molo`)" —
/// a custom type with the same name and `&mut` do not count: the former errors out, the latter leaves the
/// compile error on the user's signature;
/// the business parameter type must implement `schemars::JsonSchema` (built in for primitive types).
///
/// # Scope of applicability
///
/// The macro generates **stateless, function-like tools**: the schema is static at compile time, and execution
/// depends only on the arguments and the injected shared state. Tools that need internal state
/// (e.g. holding a registry handle, per-session deduplication) or a dynamic
/// schema (e.g. parameter constraints that change with runtime state) should implement the `Tool` trait
/// directly — the same argument definition approach (serde deserialization + schemars schema generation)
/// still applies; see `LoadSkillTool` for a precedent. (Note: `molo::` paths cannot be resolved in this
/// macro crate's docs; the identifiers above are textual references only. See the main crate docs for real paths.)
///
/// # Examples
///
/// Complete usage. This block is not compiled: the macro-generated code references `::molo::` /
/// `::serde_json::` / `::schemars::`, so prepare them as described under dependency prerequisites in a real crate.
///
/// ```text
/// use molo::tool::{SharedState, ToolError, ToolRegistry};
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// /// Define multi-field parameters as a struct; `#[schemars(description)]` on a field becomes part of the
/// /// JSON Schema, which is the main basis for the model to generate arguments.
/// #[derive(Deserialize, JsonSchema)]
/// struct WeatherArgs {
///     /// The city name.
///     #[schemars(description = "City name, e.g. \"Beijing\"")]
///     city: String,
///     /// Temperature unit, defaults to celsius.
///     #[schemars(description = "Temperature unit: celsius or fahrenheit")]
///     unit: Option<String>,
/// }
///
/// /// A tool: the struct, the parameter schema, and call are all generated by the macro; the `name`
/// /// attribute sets the registration name of the tool, defaulting to the function name.
/// #[molo::tool(description = "Query the weather for a given city", name = "get_weather")]
/// async fn fetch_weather(
///     args: WeatherArgs,
///     state: &SharedState,
/// ) -> Result<String, ToolError> {
///     // Read and write cross-tool shared data through SharedState; the type is the key.
///     let calls = state.get::<u32>().unwrap_or(0) + 1;
///     state.insert(calls);
///     let unit = args.unit.as_deref().unwrap_or("celsius");
///     Ok(format!("{}: sunny, {unit}; this tool has been called {calls} times", args.city))
/// }
///
/// // The macro-generated `FetchWeather` struct lives in the same scope as the function; register it and hand it to the Agent:
/// let mut registry = ToolRegistry::new();
/// registry.register(FetchWeather);
/// ```
///
/// Primitive-type parameters and zero parameters are supported as well: a primitive is automatically wrapped
/// in an object Schema, and zero parameters yield an empty object Schema (rules in the next section).
///
/// # Parameter JSON Schema rules
///
/// Tool arguments are always JSON objects on the wire (a model-side convention); the macro distinguishes three parameter shapes:
///
/// - A single struct parameter: its object Schema is used directly as `parameters`, with field-level
///   `#[schemars(description)]` preserved as-is;
/// - A single primitive-type parameter: wrapped in an object
///   `{ "type": "object", "properties": { "<param name>": <the type's Schema> } }`,
///   with the parameter name listed in `required`;
/// - Zero parameters: an empty object `{ "type": "object", "properties": {} }`.
///
/// On the `call` side, arguments are parsed with the same rules: object arguments deserialize into a struct as a whole; primitive-type
/// arguments take the value from the `properties.<param name>` field.
///
/// # Errors
///
/// The macro produces no runtime errors (tool errors are all collected into the `ToolError` return); for the
/// following violations it errors at compile time, pointing at the function signature:
///
/// - The function is not an `async fn`;
/// - The function has generic parameters;
/// - A `self` receiver is present (tools are stateless functions);
/// - More than one business parameter;
/// - A `&SharedState` parameter appears twice, or not at the end of the signature;
/// - A `&mut SharedState` is present (state injection uses an immutable reference);
/// - The parameter pattern is not a simple identifier (e.g. destructuring);
/// - An unknown attribute, or a missing required `description`.
/// - An invalid `risk` or `side_effects` string.
///
/// The macro does not explicitly check the return type — a type mismatch
/// surfaces as a compile error on the generated `call` method.
///
/// # Choosing between the macro and a hand-written `impl Tool`
///
/// - With the macro: the parameter Schema is derived statically from the function signature, definition and implementation are generated
///   in one place, and boilerplate stays minimal even with many tools; suited to ordinary tools whose parameter structure is known at compile time.
/// - Hand-written: full control over the `ToolSchema` contents (e.g. when the Schema comes from runtime configuration or
///   is assembled dynamically); suited to scenarios where parameters cannot be derived statically.
///
/// For simple tools with a single primitive parameter or none at all, just use the macro.
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ToolArgs);
    let item = parse_macro_input!(input as ItemFn);
    match expand_tool(args, item) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_tool(args: ToolArgs, item: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = item;

    if sig.asyncness.is_none() {
        return Err(syn::Error::new(
            sig.ident.span(),
            "#[molo::tool] can only be applied to async fns",
        ));
    }
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new(
            sig.ident.span(),
            "#[molo::tool] does not support generic functions",
        ));
    }

    // Classify parameters: at most one business parameter, error on any more; `&SharedState` is recognized
    // by its reference shape and type path (see is_shared_state) — `&mut` or a custom type with the same name
    // (e.g. `my_types::SharedState`) does not count as shared state: the former errors out directly, the latter is treated as a
    // business parameter, and a type-mismatch compile error lands on the user's signature where it can be located.
    let mut args_param: Option<(Ident, Type)> = None;
    let mut state_param = false;
    for input in &sig.inputs {
        match input {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new(
                    sig.ident.span(),
                    "#[molo::tool] does not support self (tools are stateless functions)",
                ));
            }
            FnArg::Typed(pat_type) => {
                if is_mut_shared_state(&pat_type.ty) {
                    return Err(syn::Error::new(
                        pat_type.ty.span(),
                        "&SharedState requires an immutable reference (tool state injection uses `&SharedState`, not `&mut`)",
                    ));
                }
                if is_shared_state(&pat_type.ty) {
                    if state_param {
                        return Err(syn::Error::new(
                            pat_type.ty.span(),
                            "the &SharedState parameter may appear only once",
                        ));
                    }
                    state_param = true;
                } else if args_param.is_some() {
                    return Err(syn::Error::new(
                        pat_type.ty.span(),
                        "too many parameters: at most one business parameter + optional &SharedState; \
                         define a parameter struct for multiple fields (derive JsonSchema)",
                    ));
                } else {
                    let ident = match &*pat_type.pat {
                        Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                        _ => {
                            return Err(syn::Error::new(
                                pat_type.pat.span(),
                                "parameters must be simple identifiers",
                            ));
                        }
                    };
                    args_param = Some((ident, (*pat_type.ty).clone()));
                }
            }
        }
    }

    // Position check: `&SharedState` must be at the end of the signature (the docs promise an "optional trailing" position) —
    // placing it in the middle would generate calls with misaligned arguments, and the compile error would land on generated
    // code that is hard to locate; here we give a clear error pointing at the signature instead.
    if state_param {
        let state_is_last = matches!(
            sig.inputs.last(),
            Some(FnArg::Typed(pat_type)) if is_shared_state(&pat_type.ty)
        );
        if !state_is_last {
            return Err(syn::Error::new(
                sig.span(),
                "&SharedState must be the last parameter (optional trailing position)",
            ));
        }
    }

    // Keep the user-visible visibility on the generated marker struct only.
    // The renamed function is an implementation detail and must not leak into
    // the downstream crate's public API or rustdoc output.
    let impl_fn = format_ident!("__molo_impl_{}", sig.ident);
    let struct_name = format_ident!("{}", snake_to_pascal(&sig.ident.to_string()));
    let mut new_sig = sig.clone();
    new_sig.ident = impl_fn.clone();
    let original_fn = quote! { #(#attrs)* #new_sig #block };

    let name = args.name.unwrap_or_else(|| sig.ident.to_string());
    let description = args.description;
    let protected = args.protected;
    let side_effects = args.side_effects.tokens();
    let risk = args.risk.tokens();
    let requires_confirmation = args.requires_confirmation;
    let timeout = match args.timeout_secs {
        Some(seconds) => {
            quote! { ::std::option::Option::Some(::std::time::Duration::from_secs(#seconds)) }
        }
        None => quote! { ::std::option::Option::None },
    };
    // The generated struct carries a rustdoc line (naming which function it wraps): doc comments in generated
    // code are `#[doc = "..."]` string literals, so they must be formatted before being put into quote!;
    // the function name is interpolated via sig.ident (Display yields the real name), not stringify! —
    // stringify! on the macro-definition side only yields literal tokens like `sig . ident`.
    let sig_doc = format!(
        "Generated by `#[molo::tool]`: tool implementation wrapping `{}`.",
        sig.ident
    );

    // The parameter Schema expression: defined once, shared by schema() and call(); cached in a OnceLock
    // (rebuilding the JSON Schema on every tool call would be pure waste, and tools may be called at high frequency);
    // strip `$schema` (a draft-07 meta field) and `title` at generation time — strictly validating endpoints
    // may reject them. Zero parameters yield an empty object (tool arguments are always JSON objects on the wire).
    let param_schema = match &args_param {
        Some((_, ty)) => quote! {{
            static __MOLO_PARAM_SCHEMA: ::std::sync::OnceLock<::serde_json::Value> =
                ::std::sync::OnceLock::new();
            __MOLO_PARAM_SCHEMA
                .get_or_init(|| {
                    let mut schema = ::serde_json::to_value(::schemars::schema_for!(#ty))
                        .expect("param schema must serialize");
                    if let ::serde_json::Value::Object(ref mut map) = schema {
                        map.remove("$schema");
                        map.remove("title");
                    }
                    schema
                })
                .clone()
        }},
        None => quote! {
            ::serde_json::json!({ "type": "object", "properties": {} })
        },
    };

    // parameters generation rules: struct parameters (object Schemas) pass through as-is; primitive-type parameters are
    // wrapped in an object (properties.<param name>); zero parameters yield an empty object. Tool arguments are always
    // JSON objects on the wire — without the wrapping, the model could not generate arguments in object form.
    let parameters = match &args_param {
        Some((ident, _)) => {
            let ident_str = ident.to_string();
            quote! {{
                let __molo_schema = #param_schema;
                let __molo_is_object = __molo_schema
                    .get("type")
                    .and_then(|t| t.as_str())
                    == Some("object")
                    && __molo_schema.get("properties").is_some();
                if __molo_is_object {
                    __molo_schema
                } else {
                    ::serde_json::json!({
                        "type": "object",
                        "properties": { #ident_str: __molo_schema },
                        "required": [#ident_str],
                    })
                }
            }}
        }
        None => quote! {
            ::serde_json::json!({ "type": "object", "properties": {} })
        },
    };

    // On the call side, arguments are parsed with the same rules as the schema: object arguments deserialize as a whole, primitive-type
    // arguments take the value from the properties.<param name> field; the call shape then depends on whether &SharedState is declared.
    let arg_parse = match &args_param {
        Some((ident, ty)) => {
            let ident_str = ident.to_string();
            quote! {
                let #ident: #ty = {
                    let __molo_schema = #param_schema;
                    let __molo_is_object = __molo_schema
                        .get("type")
                        .and_then(|t| t.as_str())
                        == Some("object")
                        && __molo_schema.get("properties").is_some();
                    let __molo_value = if __molo_is_object {
                        arguments
                    } else {
                        match arguments.get(#ident_str).cloned() {
                            ::std::option::Option::Some(value) => value,
                            ::std::option::Option::None => {
                                return ::std::result::Result::Err(
                                    ::molo::tool::ToolError::InvalidArguments(
                                        ::std::format!("missing field {}", #ident_str),
                                    ),
                                );
                            }
                        }
                    };
                    ::serde_json::from_value(__molo_value)?
                };
            }
        }
        None => quote! {},
    };
    let call_expr = match (&args_param, state_param) {
        (Some((ident, _)), true) => quote! { #impl_fn(#ident, state).await },
        (Some((ident, _)), false) => quote! { #impl_fn(#ident).await },
        (None, true) => quote! { #impl_fn(state).await },
        (None, false) => quote! { #impl_fn().await },
    };
    Ok(quote! {
        #original_fn

        #[doc = #sig_doc]
        #[doc = concat!("Tool name: ", #name)]
        // The generated struct is a pure marker type (the tool is itself); all common traits are zero-cost derives.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        #vis struct #struct_name;

        #[::molo::async_trait]
        impl ::molo::tool::Tool for #struct_name {
            fn schema(&self) -> ::molo::tool::ToolSchema {
                ::molo::tool::ToolSchema::new(
                    #name,
                    #description,
                    #parameters,
                )
                .with_policy(::molo::tool::ToolPolicy {
                    side_effects: #side_effects,
                    risk: #risk,
                    requires_confirmation: #requires_confirmation,
                    timeout: #timeout,
                    memory_policy: if #protected {
                        ::molo::tool::ToolMemoryPolicy::Protected
                    } else {
                        ::molo::tool::ToolMemoryPolicy::Normal
                    }
                })
            }

            async fn call(
                &self,
                arguments: ::serde_json::Value,
                context: ::molo::tool::ToolContext<'_>,
            ) -> ::std::result::Result<::molo::tool::ToolResult, ::molo::tool::ToolError> {
                let state = context.state;
                #arg_parse
                let __molo_output = #call_expr?;
                let mut __molo_result: ::molo::tool::ToolResult = __molo_output.into();
                if #protected {
                    if let ::molo::tool::ToolResult::Output(__molo_tool_output) = &mut __molo_result {
                        __molo_tool_output.memory_policy = ::molo::tool::ToolMemoryPolicy::Protected;
                    }
                }
                ::std::result::Result::Ok(__molo_result)
            }
        }
    })
}

/// Whether the type looks like `SharedState`: the last segment is `SharedState`, and the qualified path is empty
/// (bare `SharedState`) or starts with `molo` (`molo::SharedState` /
/// `::molo::tool::SharedState`). A custom type with the same name (e.g. `my_types::SharedState`)
/// does not match — the macro cannot know the canonical path of that type in the caller's crate at the definition site; relaxing
/// to a path prefix limits false positives to types whose path happens to start with `molo`.
fn shared_state_path(elem: &Type) -> bool {
    let Type::Path(path) = elem else {
        return false;
    };
    let segments = &path.path.segments;
    let Some(last) = segments.last() else {
        return false;
    };
    if last.ident != "SharedState" {
        return false;
    }
    let bare = segments.len() == 1 && path.path.leading_colon.is_none();
    bare || segments[0].ident == "molo"
}

/// Whether the parameter is a `&SharedState` reference (an immutable reference + [`shared_state_path`]).
fn is_shared_state(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    if reference.mutability.is_some() {
        return false;
    }
    shared_state_path(&reference.elem)
}

/// The `&mut SharedState` shape (only used to emit a clear compile error).
fn is_mut_shared_state(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    reference.mutability.is_some() && shared_state_path(&reference.elem)
}

/// snake_case → PascalCase: the name of the macro-generated tool struct is derived this way (e.g.
/// `fetch_weather` → `FetchWeather`); registration and use both go through this name.
fn snake_to_pascal(name: &str) -> String {
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> ToolArgs {
        ToolArgs {
            description: "test tool".into(),
            name: None,
            protected: false,
            side_effects: SideEffectAttr::Pure,
            risk: RiskAttr::Low,
            requires_confirmation: false,
            timeout_secs: None,
        }
    }

    fn err_of(item: ItemFn) -> String {
        match expand_tool(args(), item) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected the macro to error"),
        }
    }

    #[test]
    fn rejects_sync_fn() {
        let item: ItemFn = syn::parse_quote! {
            fn calculator(x: String) -> Result<String, ()> { Ok(x) }
        };
        assert!(err_of(item).contains("can only be applied to async fns"));
    }

    #[test]
    fn rejects_generic_fn() {
        let item: ItemFn = syn::parse_quote! {
            async fn calculator<T>(x: T) -> Result<String, ()> { Ok(String::new()) }
        };
        assert!(err_of(item).contains("does not support generic functions"));
    }

    #[test]
    fn rejects_self_receiver() {
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(&self) -> Result<String, ()> { Ok(String::new()) }
        };
        assert!(err_of(item).contains("does not support self"));
    }

    #[test]
    fn rejects_two_business_params() {
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(a: String, b: String) -> Result<String, ()> { Ok(a) }
        };
        assert!(err_of(item).contains("too many parameters"));
    }

    #[test]
    fn rejects_two_shared_state_params() {
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(
                _s1: &molo::tool::SharedState,
                _s2: &molo::tool::SharedState,
            ) -> Result<String, ()> { Ok(String::new()) }
        };
        assert!(err_of(item).contains("may appear only once"));
    }

    #[test]
    fn rejects_unknown_attribute() {
        let err = match syn::parse_str::<ToolArgs>(r#"description = "x", foo = "y""#) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected attribute parsing to fail"),
        };
        assert!(err.contains("unknown attribute"));
    }

    #[test]
    fn rejects_invalid_side_effects() {
        let err = match syn::parse_str::<ToolArgs>(
            r#"description = "x", side_effects = "destructive""#,
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected attribute parsing to fail"),
        };
        assert!(err.contains("invalid side_effects value"));
    }

    #[test]
    fn rejects_invalid_risk() {
        let err = match syn::parse_str::<ToolArgs>(r#"description = "x", risk = "severe""#) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected attribute parsing to fail"),
        };
        assert!(err.contains("invalid risk value"));
    }

    #[test]
    fn requires_description() {
        let err = match syn::parse_str::<ToolArgs>("name = \"calc\"") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected attribute parsing to fail"),
        };
        assert!(err.contains("missing required attribute description"));
    }

    #[test]
    fn rejects_shared_state_not_last() {
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(
                _state: &molo::tool::SharedState,
                x: String,
            ) -> Result<String, ()> { Ok(x) }
        };
        assert!(err_of(item).contains("must be the last parameter"));
    }

    #[test]
    fn rejects_mut_shared_state() {
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(_state: &mut molo::tool::SharedState) -> Result<String, ()> {
                Ok(String::new())
            }
        };
        assert!(err_of(item).contains("not `&mut`"));
    }

    #[test]
    fn custom_same_name_type_treated_as_business_param() {
        // A custom type with the same name (no molo prefix): not treated as shared state; expanded as a business parameter.
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(state: my_types::SharedState) -> Result<String, ()> {
                Ok(String::new())
            }
        };
        match expand_tool(args(), item) {
            Ok(_) => {}
            Err(e) => panic!("should not error, actual: {e}"),
        }
    }

    #[test]
    fn protected_attribute_generates_memory_policy() {
        let tool_args = match syn::parse_str::<ToolArgs>("description = \"x\", protected = true") {
            Ok(a) => a,
            Err(e) => panic!("attribute parsing failed: {e}"),
        };
        let item: ItemFn = syn::parse_quote! {
            async fn calculator(x: String) -> Result<String, ()> { Ok(x) }
        };
        let tokens = expand_tool(tool_args, item).unwrap().to_string();
        assert!(tokens.contains("ToolMemoryPolicy :: Protected"));
        // Defaults to false when protected is not declared.
        let tokens = expand_tool(args(), item_plain()).unwrap().to_string();
        assert!(tokens.contains("ToolMemoryPolicy :: Normal"));
    }

    #[test]
    fn policy_attributes_generate_tool_policy() {
        let tool_args = match syn::parse_str::<ToolArgs>(
            "description = \"x\", side_effects = \"read_only\", risk = \"medium\", requires_confirmation = true, timeout_secs = 10",
        ) {
            Ok(a) => a,
            Err(e) => panic!("attribute parsing failed: {e}"),
        };
        let tokens = expand_tool(tool_args, item_plain()).unwrap().to_string();
        assert!(tokens.contains("SideEffectLevel :: ReadOnly"));
        assert!(tokens.contains("RiskLevel :: Medium"));
        assert!(tokens.contains("requires_confirmation : true"));
        assert!(tokens.contains("Duration :: from_secs (10"));
    }

    #[test]
    fn generated_helper_is_private_even_for_public_tool() {
        let item: ItemFn = syn::parse_quote! {
            pub async fn calculator(x: String) -> Result<String, ()> { Ok(x) }
        };
        let tokens = expand_tool(args(), item).unwrap().to_string();
        assert!(tokens.contains("pub struct Calculator"));
        assert!(!tokens.contains("pub async fn __molo_impl_calculator"));
    }

    fn item_plain() -> ItemFn {
        syn::parse_quote! {
            async fn calculator(x: String) -> Result<String, ()> { Ok(x) }
        }
    }
}
