use thiserror::Error;

#[derive(Error, Debug)]
pub enum ZError {
    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("HTTP error: {0}")]
    Http(Box<ureq::Error>),

    #[error("LLM server error: {0}")]
    LlmServer(String),

    #[error("LLM response error: {0}")]
    LlmResponse(String),

    #[error("Tool call error: {0}")]
    ToolCall(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("ML error: {0}")]
    Ml(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl From<ureq::Error> for ZError {
    fn from(e: ureq::Error) -> Self {
        ZError::Http(Box::new(e))
    }
}

pub type Result<T> = std::result::Result<T, ZError>;
