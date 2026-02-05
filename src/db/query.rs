#![allow(clippy::module_name_repetitions)]

use crate::structs::{DataRow, DataSource, Result, ZError};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// JSON file-based data source
pub struct JsonDataSource {
    rows: Vec<DataRow>,
    #[allow(dead_code)]
    schema: Vec<String>,
}

impl JsonDataSource {
    /// Load from a JSON file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZError::Database(format!("Failed to read database file: {e}")))?;

        Self::from_json(&content)
    }

    /// Load from JSON string
    ///
    /// # Errors
    /// Returns error if JSON is invalid
    pub fn from_json(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| ZError::Database(format!("Failed to parse JSON: {e}")))?;

        let rows_value = if value.is_array() {
            value
        } else if let Some(rows) = value.get("rows") {
            rows.clone()
        } else if let Some(data) = value.get("data") {
            data.clone()
        } else {
            return Err(ZError::Database(
                "JSON must be an array or have 'rows'/'data' field".into(),
            ));
        };

        let raw_rows: Vec<Value> = serde_json::from_value(rows_value)
            .map_err(|e| ZError::Database(format!("Failed to parse rows: {e}")))?;

        let mut rows = Vec::new();
        let mut schema_keys = std::collections::HashSet::new();

        for (idx, raw) in raw_rows.into_iter().enumerate() {
            let obj = raw
                .as_object()
                .ok_or_else(|| ZError::Database("Each row must be an object".into()))?;

            // Get or generate ID
            let id = obj
                .get("id")
                .and_then(|v| v.as_str().map(String::from))
                .or_else(|| obj.get("id").and_then(|v| v.as_i64().map(|n| n.to_string())))
                .unwrap_or_else(|| format!("row_{idx}"));

            // Collect fields
            let mut fields = HashMap::new();
            for (key, value) in obj {
                if key != "id" {
                    schema_keys.insert(key.clone());
                    fields.insert(key.clone(), value.clone());
                }
            }

            rows.push(DataRow { id, fields });
        }

        let schema: Vec<String> = schema_keys.into_iter().collect();

        Ok(JsonDataSource { rows, schema })
    }

    /// Parse a simple filter like "field=value" or "field>value"
    #[allow(clippy::type_complexity)]
    fn parse_filter(filter: &str) -> Option<Box<dyn Fn(&DataRow) -> bool>> {
        // Try equals
        if let Some((field, value)) = filter.split_once('=') {
            let field = field.trim().to_string();
            let value = value.trim().to_string();
            return Some(Box::new(move |row: &DataRow| {
                row.fields
                    .get(&field)
                    .is_some_and(|v| match v {
                        Value::String(s) => s == &value,
                        Value::Number(n) => n.to_string() == value,
                        _ => v.to_string().trim_matches('"') == value,
                    })
            }));
        }

        // Try greater than
        if let Some((field, value)) = filter.split_once('>') {
            let field = field.trim().to_string();
            if let Ok(threshold) = value.trim().parse::<f64>() {
                return Some(Box::new(move |row: &DataRow| {
                    row.fields
                        .get(&field)
                        .and_then(Value::as_f64)
                        .is_some_and(|n| n > threshold)
                }));
            }
        }

        // Try less than
        if let Some((field, value)) = filter.split_once('<') {
            let field = field.trim().to_string();
            if let Ok(threshold) = value.trim().parse::<f64>() {
                return Some(Box::new(move |row: &DataRow| {
                    row.fields
                        .get(&field)
                        .and_then(Value::as_f64)
                        .is_some_and(|n| n < threshold)
                }));
            }
        }

        None
    }
}

impl DataSource for JsonDataSource {
    fn query(&self, filter: Option<&str>, limit: usize) -> Result<Vec<DataRow>> {
        let rows: Vec<DataRow> = if let Some(filter_str) = filter {
            if let Some(predicate) = Self::parse_filter(filter_str) {
                self.rows
                    .iter()
                    .filter(|row| predicate(row))
                    .take(limit)
                    .cloned()
                    .collect()
            } else {
                // Invalid filter, return empty
                eprintln!("Warning: could not parse filter '{filter_str}'");
                self.rows.iter().take(limit).cloned().collect()
            }
        } else {
            self.rows.iter().take(limit).cloned().collect()
        };

        Ok(rows)
    }

    fn get_row(&self, id: &str) -> Result<Option<DataRow>> {
        Ok(self.rows.iter().find(|r| r.id == id).cloned())
    }

    fn get_all_ids(&self) -> Result<Vec<String>> {
        Ok(self.rows.iter().map(|r| r.id.clone()).collect())
    }

    fn get_schema(&self) -> Result<Vec<String>> {
        Ok(self.schema.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_data_source() {
        let json = r#"[
            {"id": "1", "name": "alpha", "value": 10},
            {"id": "2", "name": "beta", "value": 20},
            {"id": "3", "name": "gamma", "value": 30}
        ]"#;

        let ds = JsonDataSource::from_json(json).expect("parse json");

        assert_eq!(ds.get_all_ids().expect("get ids").len(), 3);
        assert!(ds.get_row("2").expect("get row").is_some());
    }

    #[test]
    fn test_query_with_filter() {
        let json = r#"[
            {"id": "1", "name": "alpha", "value": 10},
            {"id": "2", "name": "beta", "value": 20},
            {"id": "3", "name": "gamma", "value": 30}
        ]"#;

        let ds = JsonDataSource::from_json(json).expect("parse json");

        let results = ds.query(Some("value>15"), 10).expect("query");
        assert_eq!(results.len(), 2);

        let results = ds.query(Some("name=beta"), 10).expect("query");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "2");
    }

    #[test]
    fn test_query_with_limit() {
        let json = r#"[
            {"id": "1", "value": 10},
            {"id": "2", "value": 20},
            {"id": "3", "value": 30}
        ]"#;

        let ds = JsonDataSource::from_json(json).expect("parse json");

        let results = ds.query(None, 2).expect("query");
        assert_eq!(results.len(), 2);
    }
}
