# Changelog

All notable user-facing changes are tracked here. This project is in 0.x:
minor releases may include breaking API changes when they serve the documented
runtime/harness architecture.

## Unreleased

### Changed

- Tightened public API hygiene against the Rust API Guidelines checklist.
- Kept `molo-cli` as a `publish = false` reference application by hiding its
  internal modules from the library surface.
- Hid macro-generated `__molo_impl_*` helper functions from downstream public
  APIs while keeping generated tool marker structs visible.
- Moved public configuration and policy structs toward constructor/accessor
  APIs instead of exposing mutable field layout.
- Removed the unnecessary `Provider` bound from `RetryProvider`'s struct
  definition.

### Added

- Added crates.io metadata for documentation and homepage links.
- Added this public changelog to the published package.
- Added serde support for serializable configuration and policy types.
- Added `Extend` and `FromIterator` implementations for `ToolRegistry` and
  `SkillRegistry`.
