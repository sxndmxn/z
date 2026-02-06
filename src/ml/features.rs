use crate::structs::{CsvData, FeatureMatrix, NormalizedFeatures, Result, ZError};

impl FeatureMatrix {
    /// Extract numeric features from CSV data
    ///
    /// # Errors
    /// Returns error if no numeric columns found
    pub fn from_csv(csv: &CsvData) -> Result<Self> {
        let numeric_cols = csv.numeric_column_indices();

        if numeric_cols.is_empty() {
            return Err(ZError::Ml("No numeric columns found".into()));
        }

        let names: Vec<String> = numeric_cols
            .iter()
            .filter_map(|&i| csv.headers.get(i).cloned())
            .collect();

        let mut data = Vec::new();
        let mut row_indices = Vec::new();

        for (row_idx, row) in csv.rows.iter().enumerate() {
            let mut features = Vec::new();
            let mut valid = true;

            for &col_idx in &numeric_cols {
                if let Some(val) = row.get(col_idx) {
                    if let Ok(num) = val.parse::<f64>() {
                        features.push(num);
                    } else {
                        valid = false;
                        break;
                    }
                } else {
                    valid = false;
                    break;
                }
            }

            if valid && features.len() == numeric_cols.len() {
                data.push(features);
                row_indices.push(row_idx);
            }
        }

        if data.is_empty() {
            return Err(ZError::Ml("No complete rows with numeric data".into()));
        }

        Ok(Self {
            names,
            data,
            row_indices,
        })
    }

    /// Normalize features using min-max scaling to [0, 1]
    #[must_use]
    pub fn normalize(&self) -> NormalizedFeatures {
        let mut mins = vec![f64::MAX; self.n_features()];
        let mut maxs = vec![f64::MIN; self.n_features()];

        // Find min/max for each feature
        for row in &self.data {
            for (i, &val) in row.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }

        // Normalize data
        let normalized_data: Vec<Vec<f64>> = self
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let range = maxs[i] - mins[i];
                        if range == 0.0 {
                            0.5 // Constant column
                        } else {
                            (val - mins[i]) / range
                        }
                    })
                    .collect()
            })
            .collect();

        NormalizedFeatures {
            names: self.names.clone(),
            data: normalized_data,
            row_indices: self.row_indices.clone(),
            mins,
            maxs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> CsvData {
        let content = "name,x,y\na,1.0,10.0\nb,2.0,20.0\nc,3.0,30.0";
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(content.as_bytes()).expect("write content");
        CsvData::from_file(file.path(), false).expect("parse csv")
    }

    #[test]
    fn test_feature_extraction() {
        let csv = create_test_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract features");

        assert_eq!(features.data.len(), 3);
        assert_eq!(features.n_features(), 2);
        assert_eq!(features.names, vec!["x", "y"]);
    }

    #[test]
    fn test_normalization() {
        let csv = create_test_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract features");
        let normalized = features.normalize();

        // First value should be 0.0, last should be 1.0
        assert!((normalized.data[0][0] - 0.0).abs() < 0.01);
        assert!((normalized.data[2][0] - 1.0).abs() < 0.01);
    }
}
