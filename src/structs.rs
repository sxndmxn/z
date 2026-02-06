//! Consolidated public types for the Z crate
//!
//! This module contains all public structs, enums, and traits used across the crate.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

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

    #[error("ML error: {0}")]
    Ml(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl From<ureq::Error> for ZError {
    fn from(e: ureq::Error) -> Self {
        Self::Http(Box::new(e))
    }
}

pub type Result<T> = std::result::Result<T, ZError>;

// ============================================================================
// Context Types
// ============================================================================

/// File types we recognize
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Text,
    Csv,
    Json,
    Markdown,
}

impl FileType {
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "csv" | "tsv" => Self::Csv,
            "json" => Self::Json,
            "md" | "markdown" => Self::Markdown,
            _ => Self::Text,
        }
    }

    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Markdown => "markdown",
        }
    }
}

/// Information about a context file
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub filename: String,
    pub file_type: FileType,
    pub size_bytes: u64,
    pub preview: String,
}

impl FileInfo {
    /// Format for display
    #[must_use]
    pub fn display(&self) -> String {
        format!(
            "{} ({}, {} bytes): {}...",
            self.filename,
            self.file_type.display_name(),
            self.size_bytes,
            self.preview.replace('\n', " ")
        )
    }
}

// ============================================================================
// CSV Types
// ============================================================================

/// Represents a parsed CSV/TSV file with headers and rows
#[derive(Debug, Clone)]
pub struct CsvData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl CsvData {
    /// Get number of rows
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get number of columns
    #[must_use]
    pub const fn col_count(&self) -> usize {
        self.headers.len()
    }

    /// Get a column as a vector of strings
    #[must_use]
    pub fn column(&self, index: usize) -> Option<Vec<&str>> {
        if index >= self.headers.len() {
            return None;
        }
        Some(
            self.rows
                .iter()
                .filter_map(|row| row.get(index).map(String::as_str))
                .collect(),
        )
    }

    /// Find columns that contain numeric data
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn numeric_column_indices(&self) -> Vec<usize> {
        (0..self.col_count())
            .filter(|&i| {
                self.column(i).is_some_and(|col| {
                    // Consider numeric if at least 50% of non-empty values parse as numbers
                    let non_empty: Vec<_> = col.iter().filter(|s| !s.is_empty()).collect();
                    if non_empty.is_empty() {
                        return false;
                    }
                    let numeric_count = non_empty
                        .iter()
                        .filter(|s| s.parse::<f64>().is_ok())
                        .count();
                    numeric_count as f64 / non_empty.len() as f64 >= 0.5
                })
            })
            .collect()
    }

}

// ============================================================================
// ML Types
// ============================================================================

/// Feature matrix extracted from CSV data
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    /// Feature names (column headers)
    pub names: Vec<String>,
    /// Row data as feature vectors
    pub data: Vec<Vec<f64>>,
    /// Original row indices (for mapping back)
    pub row_indices: Vec<usize>,
}

impl FeatureMatrix {
    /// Get number of features (columns)
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.names.len()
    }

    /// Get a feature column by index
    #[must_use]
    pub fn column(&self, index: usize) -> Option<Vec<f64>> {
        if index >= self.n_features() {
            return None;
        }
        Some(self.data.iter().map(|row| row[index]).collect())
    }

}

/// Normalized feature matrix with scaling parameters
#[derive(Debug, Clone)]
pub struct NormalizedFeatures {
    pub names: Vec<String>,
    pub data: Vec<Vec<f64>>,
    pub row_indices: Vec<usize>,
    #[allow(dead_code)]
    pub mins: Vec<f64>,
    #[allow(dead_code)]
    pub maxs: Vec<f64>,
}

impl NormalizedFeatures {
    /// Get number of samples
    #[must_use]
    pub const fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Get number of features
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.names.len()
    }

    /// Convert to flat `Vec<f64>` (row-major)
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.iter().flatten().copied().collect()
    }

}

/// Descriptive statistics for a numeric column
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub name: String,
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub iqr: f64,
}

impl ColumnStats {
    /// Detect outliers using IQR method (values outside 1.5 * IQR)
    #[must_use]
    pub fn outlier_indices(&self, values: &[f64]) -> Vec<usize> {
        let lower_bound = 1.5f64.mul_add(-self.iqr, self.q1);
        let upper_bound = 1.5f64.mul_add(self.iqr, self.q3);

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < lower_bound || v > upper_bound)
            .map(|(i, _)| i)
            .collect()
    }

    /// Format as a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}: n={}, mean={:.2}, std={:.2}, min={:.2}, Q1={:.2}, median={:.2}, Q3={:.2}, max={:.2}, IQR={:.2}",
            self.name, self.count, self.mean, self.std_dev, self.min, self.q1, self.median, self.q3, self.max, self.iqr
        )
    }
}

/// Result of K-means clustering
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Cluster assignment for each sample
    pub labels: Vec<usize>,
    /// Number of clusters
    pub k: usize,
    /// Cluster sizes
    pub sizes: Vec<usize>,
}

/// Represents an anomaly detected in the data
#[derive(Debug, Clone, Serialize)]
#[allow(clippy::struct_field_names)]
pub struct Anomaly {
    pub row_id: usize,
    pub anomaly_type: String,
    pub score: f64,
    pub details: String,
}

/// Correlation matrix between numeric features
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub names: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

/// Result of DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DbscanResult {
    pub labels: Vec<Option<usize>>,
    pub n_clusters: usize,
    pub n_noise: usize,
    pub sizes: Vec<usize>,
    pub epsilon: f64,
    pub min_points: usize,
}

/// Result of PCA dimensionality reduction
#[derive(Debug, Clone)]
pub struct PcaResult {
    pub n_components: usize,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_variance: Vec<f64>,
    pub feature_importance: Vec<(String, f64)>,
}

/// Combined result of the full analysis pipeline
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub column_stats: Vec<ColumnStats>,
    pub cluster_result: ClusterResult,
    pub dbscan_result: Option<DbscanResult>,
    pub anomalies: Vec<Anomaly>,
    pub correlation: Option<CorrelationMatrix>,
    pub pca: Option<PcaResult>,
}

// ============================================================================
// XML Types
// ============================================================================

/// Represents an element in the XML structure
#[derive(Debug, Clone)]
pub struct XmlElement {
    pub path: String,
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub text: Option<String>,
    pub depth: usize,
}

impl XmlElement {
    /// Format for display
    #[must_use]
    pub fn display(&self) -> String {
        let attrs = if self.attributes.is_empty() {
            String::new()
        } else {
            let attr_str: Vec<String> = self
                .attributes
                .iter()
                .map(|(k, v)| format!("{k}=\"{v}\""))
                .collect();
            format!(" [{}]", attr_str.join(", "))
        };

        let text_preview = self
            .text
            .as_ref()
            .map(|t| {
                let preview = if t.len() > 50 {
                    format!("{}...", &t[..50])
                } else {
                    t.clone()
                };
                format!(": \"{}\"", preview.replace('\n', "\\n"))
            })
            .unwrap_or_default();

        format!("{}{attrs}{text_preview}", self.path)
    }
}

// ============================================================================
// LLM Types
// ============================================================================

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
#[allow(clippy::struct_field_names)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

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

