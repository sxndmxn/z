use crate::error::Result;
use csv::ReaderBuilder;
use std::path::Path;

/// Represents a parsed CSV/TSV file with headers and rows
#[derive(Debug, Clone)]
pub struct CsvData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl CsvData {
    /// Parse a CSV or TSV file
    pub fn from_file(path: &Path, is_tsv: bool) -> Result<Self> {
        let delimiter = if is_tsv { b'\t' } else { b',' };

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .flexible(true)
            .from_path(path)?;

        let headers: Vec<String> = reader
            .headers()?
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut rows = Vec::new();
        for result in reader.records() {
            let record = result?;
            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            rows.push(row);
        }

        Ok(CsvData { headers, rows })
    }

    /// Get number of rows
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get number of columns
    pub fn col_count(&self) -> usize {
        self.headers.len()
    }

    /// Get column index by name
    #[allow(dead_code)]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|h| h == name)
    }

    /// Get a column as a vector of strings
    pub fn column(&self, index: usize) -> Option<Vec<&str>> {
        if index >= self.headers.len() {
            return None;
        }
        Some(
            self.rows
                .iter()
                .filter_map(|row| row.get(index).map(|s| s.as_str()))
                .collect(),
        )
    }

    /// Get numeric values from a column (skipping non-numeric)
    #[allow(dead_code)]
    pub fn numeric_column(&self, index: usize) -> Option<Vec<f64>> {
        self.column(index).map(|col| {
            col.iter()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect()
        })
    }

    /// Find columns that contain numeric data
    pub fn numeric_column_indices(&self) -> Vec<usize> {
        (0..self.col_count())
            .filter(|&i| {
                if let Some(col) = self.column(i) {
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
                } else {
                    false
                }
            })
            .collect()
    }

    /// Get a row by index
    pub fn row(&self, index: usize) -> Option<&Vec<String>> {
        self.rows.get(index)
    }

    /// Convert to a summary string for LLM context
    pub fn summary(&self) -> String {
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
            let numeric_names: Vec<_> = numeric_cols
                .iter()
                .filter_map(|&i| self.headers.get(i))
                .collect();
            let names_str: Vec<&str> = numeric_names.iter().map(|s| s.as_str()).collect();
            summary.push_str(&format!("Numeric columns: {}\n", names_str.join(", ")));
        }

        // Show first few rows as preview
        let preview_count = std::cmp::min(3, self.row_count());
        if preview_count > 0 {
            summary.push_str(&format!("\nFirst {} rows:\n", preview_count));
            for i in 0..preview_count {
                if let Some(row) = self.row(i) {
                    summary.push_str(&format!("  {}: {}\n", i + 1, row.join(", ")));
                }
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn test_parse_csv() {
        let csv_content = "name,value,count\nalpha,1.5,10\nbeta,2.5,20\ngamma,3.5,30";
        let file = create_test_csv(csv_content);

        let data = CsvData::from_file(file.path(), false).unwrap();

        assert_eq!(data.headers, vec!["name", "value", "count"]);
        assert_eq!(data.row_count(), 3);
        assert_eq!(data.col_count(), 3);
    }

    #[test]
    fn test_numeric_columns() {
        let csv_content = "name,value,count\nalpha,1.5,10\nbeta,2.5,20\ngamma,3.5,30";
        let file = create_test_csv(csv_content);

        let data = CsvData::from_file(file.path(), false).unwrap();
        let numeric = data.numeric_column_indices();

        // "value" and "count" should be numeric
        assert_eq!(numeric, vec![1, 2]);
    }

    #[test]
    fn test_get_numeric_column() {
        let csv_content = "name,value\na,1.0\nb,2.0\nc,3.0";
        let file = create_test_csv(csv_content);

        let data = CsvData::from_file(file.path(), false).unwrap();
        let values = data.numeric_column(1).unwrap();

        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
