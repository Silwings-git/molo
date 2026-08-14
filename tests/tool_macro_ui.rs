#[test]
fn tool_macro_rejects_invalid_definitions() {
    if !cfg!(feature = "macros") {
        return;
    }

    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/tool_macro/*.rs");
}
