//! LLM tool definitions and handlers for the modify phase

use crate::context::ContextManager;
use crate::error::{Result, ZError};
use crate::xml::XmlModifier;
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

/// Get the tool definitions for the modify phase
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn get_modify_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // Context tools
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "list_files".to_string(),
                description: "List all available context files with their types and previews."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "read_file".to_string(),
                description: "Read the contents of a context file. Content is truncated if large."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename to read (e.g., 'summary.txt', 'clusters.csv')"
                        }
                    },
                    "required": ["filename"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "query_csv".to_string(),
                description:
                    "Query rows from a CSV context file with optional filtering.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The CSV filename to query"
                        },
                        "filter": {
                            "type": "string",
                            "description": "Optional text filter to match rows"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum rows to return (default: 20)"
                        }
                    },
                    "required": ["filename"]
                }),
            },
        },
        // XML tools
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_xml_structure".to_string(),
                description: "Get the hierarchical structure of the XML file.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "query_xml".to_string(),
                description: "Find XML elements matching a path pattern. Supports: element, parent/child, element[@attr='value']".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Path pattern to match (e.g., 'item', 'items/item', 'item[@id=\"1\"]')"
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_element".to_string(),
                description: "Get a specific XML element by its exact path.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Exact path to the element (e.g., 'root/items/item')"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        // Modification tool
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "modify_xml".to_string(),
                description: "Modify the XML file. Operations: update_text, set_attribute, delete, insert".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["update_text", "set_attribute", "delete", "insert"],
                            "description": "The modification operation"
                        },
                        "path": {
                            "type": "string",
                            "description": "Path pattern to target element(s)"
                        },
                        "value": {
                            "type": "string",
                            "description": "New text value (for update_text) or attribute value (for set_attribute)"
                        },
                        "attr_name": {
                            "type": "string",
                            "description": "Attribute name (for set_attribute)"
                        },
                        "element_name": {
                            "type": "string",
                            "description": "Name of new element (for insert)"
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Attributes for new element (for insert)"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text content for new element (for insert)"
                        }
                    },
                    "required": ["operation", "path"]
                }),
            },
        },
        // Completion tool
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "finish".to_string(),
                description: "Signal that all modifications are complete.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of modifications made"
                        }
                    },
                    "required": ["summary"]
                }),
            },
        },
    ]
}

/// Tool handler for the modify phase
pub struct ModifyToolHandler<'a> {
    context: &'a ContextManager,
    xml: &'a XmlModifier,
    modifications: Vec<String>,
    finished: bool,
}

impl<'a> ModifyToolHandler<'a> {
    #[must_use]
    pub fn new(context: &'a ContextManager, xml: &'a XmlModifier) -> Self {
        ModifyToolHandler {
            context,
            xml,
            modifications: Vec::new(),
            finished: false,
        }
    }

    /// Execute a tool call and return the result
    ///
    /// # Errors
    /// Returns error if tool execution fails
    pub fn execute(&mut self, tool_call: &ToolCall) -> Result<ToolResult> {
        let args: Value =
            serde_json::from_str(&tool_call.function.arguments).unwrap_or(json!({}));

        let content = match tool_call.function.name.as_str() {
            "list_files" => self.handle_list_files(),
            "read_file" => self.handle_read_file(&args)?,
            "query_csv" => self.handle_query_csv(&args)?,
            "get_xml_structure" => self.handle_get_xml_structure()?,
            "query_xml" => self.handle_query_xml(&args)?,
            "get_element" => self.handle_get_element(&args)?,
            "modify_xml" => self.handle_modify_xml(&args)?,
            "finish" => self.handle_finish(&args),
            name => return Err(ZError::ToolCall(format!("Unknown tool: {name}"))),
        };

        Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content,
        })
    }

    /// Check if finished signal was received
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get list of modifications made
    #[must_use]
    pub fn get_modifications(&self) -> &[String] {
        &self.modifications
    }

    fn handle_list_files(&self) -> String {
        use std::fmt::Write as _;

        let files = self.context.list_files();
        let mut output = format!("Available files ({}):\n", files.len());

        for info in files {
            let _ = writeln!(output, "- {}", info.display());
        }

        output
    }

    fn handle_read_file(&self, args: &Value) -> Result<String> {
        let filename = args
            .get("filename")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing filename parameter".into()))?;

        self.context.read_file(filename)
    }

    fn handle_query_csv(&self, args: &Value) -> Result<String> {
        let filename = args
            .get("filename")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing filename parameter".into()))?;

        let filter = args.get("filter").and_then(Value::as_str);
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok());

        self.context.query_csv(filename, filter, limit)
    }

    fn handle_get_xml_structure(&self) -> Result<String> {
        use std::fmt::Write as _;

        let elements = self.xml.get_structure()?;
        let mut output = String::from("XML Structure:\n");

        for elem in elements.iter().take(crate::xml::modifier::MAX_XML_ELEMENTS) {
            let indent = "  ".repeat(elem.depth);
            let _ = writeln!(output, "{indent}{}", elem.display());
        }

        if elements.len() > crate::xml::modifier::MAX_XML_ELEMENTS {
            let _ = writeln!(
                output,
                "... and {} more elements",
                elements.len() - crate::xml::modifier::MAX_XML_ELEMENTS
            );
        }

        Ok(output)
    }

    fn handle_query_xml(&self, args: &Value) -> Result<String> {
        use std::fmt::Write as _;

        let pattern = args
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing pattern parameter".into()))?;

        let elements = self.xml.query(pattern)?;

        if elements.is_empty() {
            return Ok(format!("No elements matching '{pattern}'"));
        }

        let mut output = format!("Found {} element(s) matching '{}':\n", elements.len(), pattern);
        for elem in &elements {
            let _ = writeln!(output, "- {}", elem.display());
        }

        Ok(output)
    }

    fn handle_get_element(&self, args: &Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing path parameter".into()))?;

        match self.xml.get_element(path)? {
            Some(elem) => Ok(format!("Element: {}", elem.display())),
            None => Ok(format!("No element at path '{path}'")),
        }
    }

    fn handle_modify_xml(&mut self, args: &Value) -> Result<String> {
        let operation = args
            .get("operation")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing operation parameter".into()))?;

        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing path parameter".into()))?;

        let result = match operation {
            "update_text" => self.handle_update_text(args, path)?,
            "set_attribute" => self.handle_set_attribute(args, path)?,
            "delete" => self.handle_delete(path)?,
            "insert" => self.handle_insert(args, path)?,
            _ => return Err(ZError::ToolCall(format!("Unknown operation: {operation}"))),
        };

        Ok(result)
    }

    fn handle_update_text(&mut self, args: &Value, path: &str) -> Result<String> {
        let value = args
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing value for update_text".into()))?;

        let modified = self.xml.update_text(path, value)?;
        if modified {
            self.modifications
                .push(format!("update_text: {path} = '{value}'"));
            Ok("Text updated successfully".to_string())
        } else {
            Ok("No matching element found".to_string())
        }
    }

    fn handle_set_attribute(&mut self, args: &Value, path: &str) -> Result<String> {
        let attr_name = args
            .get("attr_name")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing attr_name for set_attribute".into()))?;

        let value = args
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing value for set_attribute".into()))?;

        let modified = self.xml.set_attribute(path, attr_name, value)?;
        if modified {
            self.modifications
                .push(format!("set_attribute: {path} @{attr_name} = '{value}'"));
            Ok("Attribute set successfully".to_string())
        } else {
            Ok("No matching element found".to_string())
        }
    }

    fn handle_delete(&mut self, path: &str) -> Result<String> {
        let modified = self.xml.delete_element(path)?;
        if modified {
            self.modifications.push(format!("delete: {path}"));
            Ok("Element deleted successfully".to_string())
        } else {
            Ok("No matching element found".to_string())
        }
    }

    fn handle_insert(&mut self, args: &Value, path: &str) -> Result<String> {
        let element_name = args
            .get("element_name")
            .and_then(Value::as_str)
            .ok_or_else(|| ZError::ToolCall("Missing element_name for insert".into()))?;

        let text = args.get("text").and_then(Value::as_str);

        let attributes: Vec<(String, String)> = args
            .get("attributes")
            .and_then(Value::as_object)
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let modified = self.xml.insert_element(path, element_name, &attributes, text)?;
        if modified {
            self.modifications
                .push(format!("insert: {path} -> <{element_name}>"));
            Ok("Element inserted successfully".to_string())
        } else {
            Ok("No matching parent element found".to_string())
        }
    }

    fn handle_finish(&mut self, args: &Value) -> String {
        self.finished = true;
        let summary = args
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or("Modifications complete");

        format!(
            "Finished: {summary}\nTotal modifications: {}",
            self.modifications.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definitions() {
        let tools = get_modify_tool_definitions();
        assert!(!tools.is_empty());

        let names: Vec<_> = tools.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"list_files"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"query_csv"));
        assert!(names.contains(&"get_xml_structure"));
        assert!(names.contains(&"query_xml"));
        assert!(names.contains(&"modify_xml"));
        assert!(names.contains(&"finish"));
    }
}
