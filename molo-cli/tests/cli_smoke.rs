use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_molo")
}

fn temp_root(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("molo-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn help_prints_command_surface() {
    let output = Command::new(bin()).arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("molo reference CLI"));
    assert!(stdout.contains("molo [GLOBAL_OPTIONS] code"));
}

#[test]
fn config_check_json_redacts_api_key_value() {
    let root = temp_root("config");
    let sessions = root.join("sessions");
    let output = Command::new(bin())
        .args([
            "--workspace",
            root.to_str().unwrap(),
            "--session-dir",
            sessions.to_str().unwrap(),
            "--provider",
            "openai",
            "--api-key-env",
            "MOLO_CLI_FAKE_SECRET",
            "config",
            "check",
            "--json",
        ])
        .env("MOLO_CLI_FAKE_SECRET", "secret-value")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("MOLO_CLI_FAKE_SECRET"));
    assert!(!stdout.contains("secret-value"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn fake_chat_creates_session() {
    let root = temp_root("chat");
    let sessions = root.join("sessions");
    let output = Command::new(bin())
        .args([
            "--workspace",
            root.to_str().unwrap(),
            "--session-dir",
            sessions.to_str().unwrap(),
            "chat",
            "--no-stream",
            "hello",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fake provider response"));
    assert!(stdout.contains("session:"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn fake_code_patch_runs_through_binary() {
    let root = temp_root("code");
    let sessions = root.join("sessions");
    let output = Command::new(bin())
        .args([
            "--workspace",
            root.to_str().unwrap(),
            "--session-dir",
            sessions.to_str().unwrap(),
            "--non-interactive",
            "code",
            "--json",
            "fake-patch",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(root.join("molo-cli-fake-output.txt").exists());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"session_id\""));
    let _ = std::fs::remove_dir_all(root);
}
