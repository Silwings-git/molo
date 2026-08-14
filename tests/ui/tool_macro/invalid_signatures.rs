#![allow(dead_code)]

use molo::tool::{SharedState, ToolError};

#[molo::tool(description = "sync")]
fn sync_fn() -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "generic")]
async fn generic_fn<T>(_value: T) -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "self receiver")]
async fn self_receiver(&self) -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "too many")]
async fn too_many_business_params(_left: String, _right: String) -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "state not last")]
async fn shared_state_not_last(_state: &SharedState, _value: String) -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "mutable state")]
async fn mutable_shared_state(_state: &mut SharedState) -> Result<String, ToolError> {
    Ok(String::new())
}

#[molo::tool(description = "destructure")]
async fn destructured_param((_left, _right): (String, String)) -> Result<String, ToolError> {
    Ok(String::new())
}

fn main() {}
