use crate::db::DataSource;
use crate::error::{Result, ZError};
use crate::llm::server::LlamaServer;
use crate::llm::tools::{get_tool_definitions, ToolCall, ToolHandler};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Token usage from API response
#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

/// Response from the LLM
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

/// LLM client for conversation with tool calling
pub struct LlmClient<'a> {
    server: &'a LlamaServer,
    messages: Vec<Message>,
    max_turns: usize,
    total_usage: Usage,
}

impl<'a> LlmClient<'a> {
    pub fn new(server: &'a LlamaServer, system_prompt: &str, max_turns: usize) -> Self {
        let messages = vec![Message {
            role: "system".to_string(),
            content: Some(system_prompt.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];

        LlmClient {
            server,
            messages,
            max_turns,
            total_usage: Usage::default(),
        }
    }

    /// Get total token usage
    pub fn total_usage(&self) -> Usage {
        self.total_usage
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(Message {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Run the conversation loop with tool calling
    pub fn run_conversation(&mut self, data_source: &dyn DataSource) -> Result<Vec<String>> {
        let mut handler = ToolHandler::new(data_source);
        let tools = get_tool_definitions();

        for turn in 0..self.max_turns {
            eprintln!("LLM turn {}/{}...", turn + 1, self.max_turns);

            // Make request to LLM
            let response = self.send_request(&tools)?;

            // Check for tool calls
            if let Some(tool_calls) = &response.tool_calls {
                // Add assistant message with tool calls
                self.messages.push(Message {
                    role: "assistant".to_string(),
                    content: response.content.clone(),
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                });

                // Execute each tool call
                for tool_call in tool_calls {
                    eprintln!("  Tool call: {}", tool_call.function.name);
                    let result = handler.execute(tool_call)?;

                    // Add tool result message
                    self.messages.push(Message {
                        role: "tool".to_string(),
                        content: Some(result.content),
                        tool_calls: None,
                        tool_call_id: Some(result.tool_call_id),
                    });
                }

                // Check if selection is complete
                if handler.has_selection() {
                    eprintln!("Selection complete");
                    return Ok(handler.get_selected_rows().to_vec());
                }
            } else {
                // No tool calls - LLM finished
                eprintln!("LLM finished without selection");
                if let Some(content) = &response.content {
                    eprintln!("Final response: {}", content);
                }
                break;
            }
        }

        // Return whatever was selected (may be empty)
        Ok(handler.get_selected_rows().to_vec())
    }

    /// Send a request to the LLM
    fn send_request(&mut self, tools: &[crate::llm::tools::ToolDefinition]) -> Result<ResponseMessage> {
        let body = json!({
            "model": "default",
            "messages": self.messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.7,
            "max_tokens": 2048
        });

        let response = ureq::post(&self.server.completions_url())
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(120))
            .send_json(&body)
            .map_err(|e| ZError::Http(e))?;

        let chat_response: ChatResponse = response
            .into_json()
            .map_err(|e| ZError::LlmResponse(format!("Failed to parse response: {}", e)))?;

        // Accumulate usage
        if let Some(usage) = chat_response.usage {
            self.total_usage.prompt_tokens += usage.prompt_tokens;
            self.total_usage.completion_tokens += usage.completion_tokens;
            self.total_usage.total_tokens += usage.total_tokens;
        }

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| ZError::LlmResponse("No choices in response".into()))
    }
}

/// Build the system prompt with ML analysis
pub fn build_system_prompt(ml_summary: &str, csv_summary: &str) -> String {
    format!(
        r#"You are an AI assistant that analyzes data and selects rows to add to an XML file.

## Your Task
Based on the ML analysis and CSV data summary below, use the available tools to:
1. Query the database to see what rows are available
2. Analyze which rows would be most valuable to add based on the patterns found
3. Select the final rows using the select_rows tool

## ML Analysis Summary
{}

## CSV Data Summary
{}

## Guidelines
- Focus on selecting rows that complement the existing data
- Consider outliers and cluster membership when making decisions
- Prefer rows that fill gaps in the current data distribution
- Be concise in your reasoning to save tokens
- Call select_rows when you've made your final decision

Start by querying the database to see available options."#,
        ml_summary, csv_summary
    )
}
