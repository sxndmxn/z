//! Correlation matrix computation

use crate::structs::{CorrelationMatrix, FeatureMatrix, Result};
use crate::ml::stats::correlation;

/// Compute the `NxN` correlation matrix between all numeric features
///
/// # Errors
/// Returns error if feature extraction or correlation calculation fails
pub fn correlation_matrix(features: &FeatureMatrix) -> Result<CorrelationMatrix> {
    let n = features.n_features();
    let mut matrix = vec![vec![0.0; n]; n];

    let columns: Vec<Vec<f64>> = (0..n)
        .filter_map(|i| features.column(i))
        .collect();

    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let r = correlation(&columns[i], &columns[j])?;
            matrix[i][j] = r;
            matrix[j][i] = r;
        }
    }

    Ok(CorrelationMatrix {
        names: features.names.clone(),
        matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::CsvData;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_correlation_matrix() {
        let content = "a,b,c\n1.0,2.0,10.0\n2.0,4.0,20.0\n3.0,6.0,30.0";
        let mut file = NamedTempFile::new().expect("create");
        file.write_all(content.as_bytes()).expect("write");

        let csv = CsvData::from_file(file.path(), false).expect("parse");
        let features = FeatureMatrix::from_csv(&csv).expect("extract");
        let corr = correlation_matrix(&features).expect("correlate");

        assert_eq!(corr.names.len(), 3);
        assert_eq!(corr.matrix.len(), 3);
        // Diagonal should be 1.0
        assert!((corr.matrix[0][0] - 1.0).abs() < 0.01);
        // a and b are perfectly correlated
        assert!((corr.matrix[0][1] - 1.0).abs() < 0.01);
    }
}
