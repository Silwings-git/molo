#![allow(dead_code)]

use molo::tool::ToolError;

#[molo::tool(description = "wrong return")]
async fn wrong_return_type() -> Result<u32, ToolError> {
    Ok(42)
}

fn main() {}
