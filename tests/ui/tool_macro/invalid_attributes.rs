#![allow(dead_code)]

use molo::tool::ToolError;

#[molo::tool(name = "missing_description")]
async fn missing_description() -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "unknown", unknown = "value")]
async fn unknown_attribute() -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "bad risk", risk = "severe")]
async fn invalid_risk() -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "bad side effects", side_effects = "destructive")]
async fn invalid_side_effects() -> Result<String, ToolError> {
    Ok(String::new())
}

fn main() {}
