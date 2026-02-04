use crate::db::DataSource;
use crate::error::{Result, ZError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Tool definition for LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A tool call from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Result of a tool execution
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

/// Get the tool definitions for the LLM
#[must_use]
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "query_database".to_string(),
                description: "Query available rows that can be added to XML. Returns matching rows from the database.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Optional filter criteria (e.g., 'category=books' or 'price>100')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return (default: 10)"
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "select_rows".to_string(),
                description: "Select specific rows to add to the XML file. Call this when you've decided which rows to add.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "row_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "List of row IDs to select for adding to XML"
                        }
                    },
                    "required": ["row_ids"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_row_details".to_string(),
                description: "Get detailed information about specific rows by their IDs.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "row_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "List of row IDs to get details for"
                        }
                    },
                    "required": ["row_ids"]
                }),
            },
        },
    ]
}

/// Tool handler that processes tool calls
pub struct ToolHandler<'a> {
    data_source: &'a dyn DataSource,
    selected_rows: Vec<String>,
}

impl<'a> ToolHandler<'a> {
    #[must_use]
    pub fn new(data_source: &'a dyn DataSource) -> Self {
        ToolHandler {
            data_source,
            selected_rows: Vec::new(),
        }
    }

    /// Execute a tool call and return the result
    ///
    /// # Errors
    /// Returns error if tool execution fails
    pub fn execute(&mut self, tool_call: &ToolCall) -> Result<ToolResult> {
        let args: Value = serde_json::from_str(&tool_call.function.arguments)
            .unwrap_or(json!({}));

        let content = match tool_call.function.name.as_str() {
            "query_database" => self.handle_query_database(&args)?,
            "select_rows" => self.handle_select_rows(&args)?,
            "get_row_details" => self.handle_get_row_details(&args)?,
            name => return Err(ZError::ToolCall(format!("Unknown tool: {name}"))),
        };

        Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content,
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn handle_query_database(&self, args: &Value) -> Result<String> {
        let filter = args.get("filter").and_then(Value::as_str);
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(10) as usize;

        let rows = self.data_source.query(filter, limit)?;

        // Format as compact JSON for token efficiency
        let result = json!({
            "count": rows.len(),
            "rows": rows
        });

        Ok(serde_json::to_string(&result)?)
    }

    fn handle_select_rows(&mut self, args: &Value) -> Result<String> {
        let row_ids: Vec<String> = args
            .get("row_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if row_ids.is_empty() {
            return Err(ZError::ToolCall("No row IDs provided".into()));
        }

        // Validate that rows exist
        for id in &row_ids {
            if self.data_source.get_row(id)?.is_none() {
                return Err(ZError::ToolCall(format!("Row not found: {id}")));
            }
        }

        self.selected_rows.clone_from(&row_ids);

        Ok(json!({
            "status": "success",
            "selected_count": row_ids.len(),
            "row_ids": row_ids
        })
        .to_string())
    }

    fn handle_get_row_details(&self, args: &Value) -> Result<String> {
        let row_ids: Vec<String> = args
            .get("row_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut details = Vec::new();
        for id in &row_ids {
            if let Some(row) = self.data_source.get_row(id)? {
                details.push(row);
            }
        }

        Ok(serde_json::to_string(&details)?)
    }

    /// Get the selected rows after the conversation
    #[must_use]
    pub fn get_selected_rows(&self) -> &[String] {
        &self.selected_rows
    }

    /// Check if selection is complete
    #[must_use]
    pub fn has_selection(&self) -> bool {
        !self.selected_rows.is_empty()
    }
}
