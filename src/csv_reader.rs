#![allow(clippy::module_name_repetitions)]

use crate::structs::{CsvData, Result};
use csv::ReaderBuilder;
use std::path::Path;

impl CsvData {
    /// Parse a CSV or TSV file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn from_file(path: &Path, is_tsv: bool) -> Result<Self> {
        let delimiter = if is_tsv { b'\t' } else { b',' };

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .flexible(true)
            .from_path(path)?;

        let headers: Vec<String> = reader.headers()?.iter().map(String::from).collect();

        let mut rows = Vec::new();
        for result in reader.records() {
            let record = result?;
            let row: Vec<String> = record.iter().map(String::from).collect();
            rows.push(row);
        }

        Ok(Self { headers, rows })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(content.as_bytes()).expect("write content");
        file
    }

    #[test]
    fn test_parse_csv() {
        let csv_content = "name,value,count\nalpha,1.5,10\nbeta,2.5,20\ngamma,3.5,30";
        let file = create_test_csv(csv_content);

        let data = CsvData::from_file(file.path(), false).expect("parse csv");

        assert_eq!(data.headers, vec!["name", "value", "count"]);
        assert_eq!(data.row_count(), 3);
        assert_eq!(data.col_count(), 3);
    }

    #[test]
    fn test_numeric_columns() {
        let csv_content = "name,value,count\nalpha,1.5,10\nbeta,2.5,20\ngamma,3.5,30";
        let file = create_test_csv(csv_content);

        let data = CsvData::from_file(file.path(), false).expect("parse csv");
        let numeric = data.numeric_column_indices();

        // "value" and "count" should be numeric
        assert_eq!(numeric, vec![1, 2]);
    }

}
