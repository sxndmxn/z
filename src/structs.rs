//! Consolidated public types for the Z crate
//!
//! This module contains all public structs, enums, and traits used across the crate.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
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

    #[allow(dead_code)]
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
    pub fn display_name(self) -> &'static str {
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
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get number of columns
    #[must_use]
    pub fn col_count(&self) -> usize {
        self.headers.len()
    }

    /// Get column index by name
    #[allow(dead_code)]
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
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

    /// Get numeric values from a column (skipping non-numeric)
    #[allow(dead_code)]
    #[must_use]
    pub fn numeric_column(&self, index: usize) -> Option<Vec<f64>> {
        self.column(index).map(|col| {
            col.iter()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect()
        })
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

    /// Get a row by index
    #[allow(dead_code)]
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&Vec<String>> {
        self.rows.get(index)
    }

    /// Convert to a summary string for LLM context
    #[allow(dead_code)]
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write as _;

        let mut summary = format!(
            "CSV Data: {} rows x {} columns\n",
            self.row_count(),
            self.col_count()
        );
        summary.push_str("Columns: ");
        summary.push_str(&self.headers.join(", "));
        summary.push('\n');

        let numeric_cols = self.numeric_column_indices();
        if !numeric_cols.is_empty() {
            let numeric_names: Vec<&str> = numeric_cols
                .iter()
                .filter_map(|&i| self.headers.get(i).map(String::as_str))
                .collect();
            let _ = writeln!(summary, "Numeric columns: {}", numeric_names.join(", "));
        }

        // Show first few rows as preview
        let preview_count = std::cmp::min(3, self.row_count());
        if preview_count > 0 {
            let _ = writeln!(summary, "\nFirst {preview_count} rows:");
            for i in 0..preview_count {
                if let Some(row) = self.row(i) {
                    let _ = writeln!(summary, "  {}: {}", i + 1, row.join(", "));
                }
            }
        }

        summary
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
    /// Get number of samples (rows)
    #[allow(dead_code)]
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Get number of features (columns)
    #[must_use]
    pub fn n_features(&self) -> usize {
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

    /// Convert to flat `Vec<f64>` (row-major)
    #[allow(dead_code)]
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.iter().flatten().copied().collect()
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
    pub fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Get number of features
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.names.len()
    }

    /// Convert to flat `Vec<f64>` (row-major)
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.iter().flatten().copied().collect()
    }

    /// Denormalize a single value
    #[allow(dead_code)]
    #[must_use]
    pub fn denormalize(&self, feature_idx: usize, normalized_val: f64) -> f64 {
        let range = self.maxs[feature_idx] - self.mins[feature_idx];
        self.mins[feature_idx] + normalized_val * range
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
        let lower_bound = self.q1 - 1.5 * self.iqr;
        let upper_bound = self.q3 + 1.5 * self.iqr;

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
    #[allow(dead_code)]
    pub labels: Vec<usize>,
    /// Number of clusters
    pub k: usize,
    /// Cluster sizes
    pub sizes: Vec<usize>,
    /// Original row indices for each cluster
    pub cluster_members: Vec<Vec<usize>>,
}

impl ClusterResult {
    /// Get summary for LLM context
    #[allow(dead_code)]
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write as _;

        let mut s = format!("K-means clustering with k={}\n", self.k);
        for (i, size) in self.sizes.iter().enumerate() {
            let _ = writeln!(s, "  Cluster {i}: {size} samples");
        }
        s
    }

    /// Get row indices for a specific cluster
    #[allow(dead_code)]
    #[must_use]
    pub fn get_cluster(&self, cluster_id: usize) -> Option<&Vec<usize>> {
        self.cluster_members.get(cluster_id)
    }
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

// ============================================================================
// Database Types
// ============================================================================

/// A row in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRow {
    pub id: String,
    #[serde(flatten)]
    pub fields: HashMap<String, Value>,
}

/// Trait for data sources that can be queried
#[allow(dead_code)]
pub trait DataSource: Send + Sync {
    /// Query rows with optional filter and limit
    ///
    /// # Errors
    /// Returns error if query fails
    fn query(&self, filter: Option<&str>, limit: usize) -> Result<Vec<DataRow>>;

    /// Get a specific row by ID
    ///
    /// # Errors
    /// Returns error if lookup fails
    fn get_row(&self, id: &str) -> Result<Option<DataRow>>;

    /// Get all available row IDs
    ///
    /// # Errors
    /// Returns error if retrieval fails
    fn get_all_ids(&self) -> Result<Vec<String>>;

    /// Get schema/column information
    ///
    /// # Errors
    /// Returns error if schema retrieval fails
    #[allow(dead_code)]
    fn get_schema(&self) -> Result<Vec<String>>;
}
