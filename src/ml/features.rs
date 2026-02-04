use crate::csv_reader::CsvData;
use crate::error::{Result, ZError};

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
    /// Extract numeric features from CSV data
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

        Ok(FeatureMatrix {
            names,
            data,
            row_indices,
        })
    }

    /// Get number of samples (rows)
    #[allow(dead_code)]
    pub fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Get number of features (columns)
    pub fn n_features(&self) -> usize {
        self.names.len()
    }

    /// Get a feature column by index
    pub fn column(&self, index: usize) -> Option<Vec<f64>> {
        if index >= self.n_features() {
            return None;
        }
        Some(self.data.iter().map(|row| row[index]).collect())
    }

    /// Normalize features using min-max scaling to [0, 1]
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

    /// Convert to flat Vec<f64> (row-major)
    #[allow(dead_code)]
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
    pub fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.names.len()
    }

    /// Convert to flat Vec<f64> (row-major)
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.iter().flatten().copied().collect()
    }

    /// Denormalize a single value
    #[allow(dead_code)]
    pub fn denormalize(&self, feature_idx: usize, normalized_val: f64) -> f64 {
        let range = self.maxs[feature_idx] - self.mins[feature_idx];
        self.mins[feature_idx] + normalized_val * range
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> CsvData {
        let content = "name,x,y\na,1.0,10.0\nb,2.0,20.0\nc,3.0,30.0";
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        CsvData::from_file(file.path(), false).unwrap()
    }

    #[test]
    fn test_feature_extraction() {
        let csv = create_test_csv();
        let features = FeatureMatrix::from_csv(&csv).unwrap();

        assert_eq!(features.n_samples(), 3);
        assert_eq!(features.n_features(), 2);
        assert_eq!(features.names, vec!["x", "y"]);
    }

    #[test]
    fn test_normalization() {
        let csv = create_test_csv();
        let features = FeatureMatrix::from_csv(&csv).unwrap();
        let normalized = features.normalize();

        // First value should be 0.0, last should be 1.0
        assert!((normalized.data[0][0] - 0.0).abs() < 0.01);
        assert!((normalized.data[2][0] - 1.0).abs() < 0.01);
    }
}
