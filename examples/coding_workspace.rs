use molo::{
    FileReadOptions, FileWriteContent, ListFilesQuery, LocalWorkspace, Workspace, WorkspacePath,
    WriteFileRequest,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::current_dir()?;
    let workspace = LocalWorkspace::new(root)?;
    let path = WorkspacePath::parse("target/molo-coding-workspace-example.txt")?;

    workspace
        .write_file(WriteFileRequest {
            path: path.clone(),
            content: FileWriteContent::Text("hello from molo coding\n".to_string()),
            expected_version: None,
            create: true,
            overwrite: true,
        })
        .await?;

    let content = workspace
        .read_file(
            &path,
            FileReadOptions {
                max_bytes: Some(1024),
                include_binary: false,
            },
        )
        .await?;
    println!("read {} bytes from {}", content.version.len, path.display());

    let entries = workspace
        .list_files(ListFilesQuery {
            path: WorkspacePath::parse("target")?,
            recursive: false,
            max_entries: Some(10),
            include_hidden: false,
            respect_gitignore: true,
        })
        .await?;
    println!("target entries returned: {}", entries.len());
    Ok(())
}
