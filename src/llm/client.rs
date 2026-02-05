use crate::context::ContextManager;
use crate::llm::server::LlamaServer;
use crate::llm::tools::{get_modify_tool_definitions, ModifyToolHandler};
use crate::structs::{Message, Result, ToolCall, ToolDefinition, Usage, ZError};
use crate::xml::XmlModifier;
use serde::Deserialize;
use serde_json::json;

/// Response from the LLM (private)
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
    #[must_use]
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
    #[must_use]
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

    /// Run the modify conversation loop with tool calling
    ///
    /// # Errors
    /// Returns error if LLM communication fails
    pub fn run_modify_conversation(
        &mut self,
        context: &ContextManager,
        xml: &XmlModifier,
    ) -> Result<Vec<String>> {
        let mut handler = ModifyToolHandler::new(context, xml);
        let tools = get_modify_tool_definitions();

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
                    eprintln!("  Tool: {}()", tool_call.function.name);
                    let result = handler.execute(tool_call)?;

                    // Add tool result message
                    self.messages.push(Message {
                        role: "tool".to_string(),
                        content: Some(result.content),
                        tool_calls: None,
                        tool_call_id: Some(result.tool_call_id),
                    });
                }

                // Check if finished
                if handler.is_finished() {
                    eprintln!("LLM signaled completion");
                    return Ok(handler.get_modifications().to_vec());
                }
            } else {
                // No tool calls - LLM finished without using finish tool
                eprintln!("LLM finished (no more tool calls)");
                if let Some(content) = &response.content {
                    eprintln!("Final response: {content}");
                }
                break;
            }
        }

        // Return whatever modifications were made
        Ok(handler.get_modifications().to_vec())
    }

    /// Send a request to the LLM
    fn send_request(&mut self, tools: &[ToolDefinition]) -> Result<ResponseMessage> {
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
            .send_json(&body)?;

        let chat_response: ChatResponse = response
            .into_json()
            .map_err(|e| ZError::LlmResponse(format!("Failed to parse response: {e}")))?;

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

/// Build a minimal system prompt for the modify phase (~400 tokens)
#[must_use]
pub fn build_modify_system_prompt(context: &ContextManager) -> String {
    let file_index = context.build_file_index_summary();

    format!(
        r"You are an AI that modifies XML files based on ML analysis results.

## Available Context Files
{file_index}
## Your Task
1. Read context files to understand the ML analysis
2. Explore the XML structure
3. Decide what modifications to make based on analysis + any instructions
4. Execute modifications using modify_xml tool
5. Call finish when done

## Tools
- list_files: See available context files
- read_file: Read a file's content
- query_csv: Filter/search CSV rows
- get_xml_structure: See XML hierarchy
- query_xml: Find elements by pattern
- get_element: Get specific element
- modify_xml: Insert/update/delete elements
- finish: Signal completion

Start by reading summary.txt to understand the analysis."
    )
}
