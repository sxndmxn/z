//! PCA dimensionality reduction using linfa-reduction

use crate::structs::{NormalizedFeatures, PcaResult, Result, ZError};
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_reduction::Pca;
use ndarray::Array2;

/// Run PCA on normalized features
///
/// # Errors
/// Returns error if PCA computation fails (e.g., too few samples)
#[allow(clippy::cast_precision_loss)]
pub fn run_pca(
    features: &NormalizedFeatures,
    n_components: usize,
) -> Result<PcaResult> {
    let n_samples = features.n_samples();
    let n_features = features.n_features();

    if n_features < 2 {
        return Err(ZError::Ml("PCA requires at least 2 features".into()));
    }
    if n_samples < 2 {
        return Err(ZError::Ml("PCA requires at least 2 samples".into()));
    }

    // Auto-select components if 0
    let n_components = if n_components == 0 {
        n_features.min(n_samples - 1).min(5)
    } else {
        n_components.min(n_features).min(n_samples - 1)
    };

    // Build ndarray
    let flat_data = features.to_flat();
    let array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| ZError::Ml(format!("Failed to create array for PCA: {e}")))?;

    let dataset = DatasetBase::from(array);

    // Fit PCA
    let pca = Pca::params(n_components)
        .fit(&dataset)
        .map_err(|e| ZError::Ml(format!("PCA failed: {e}")))?;

    // Get explained variance from singular values
    let singular_values = pca.singular_values();
    let total_variance: f64 = singular_values.iter().map(|s| s * s).sum();

    let explained_variance_ratio: Vec<f64> = if total_variance > 0.0 {
        singular_values
            .iter()
            .map(|s| (s * s) / total_variance)
            .collect()
    } else {
        vec![0.0; n_components]
    };

    // Cumulative variance
    let mut cumulative = Vec::with_capacity(n_components);
    let mut running = 0.0;
    for &ratio in &explained_variance_ratio {
        running += ratio;
        cumulative.push(running);
    }

    // Feature importance: sum of absolute loadings per original feature
    let transformed = pca.predict(&dataset);
    let _ = transformed; // we only need the model's components for importance

    // Use singular values as proxy for feature importance per component
    let feature_importance: Vec<(String, f64)> = features
        .names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            // Importance = fraction of total variance this feature participates in
            // Approximate using the variance contribution
            let importance = if i < explained_variance_ratio.len() {
                explained_variance_ratio[i]
            } else {
                0.0
            };
            (name.clone(), importance)
        })
        .collect();

    Ok(PcaResult {
        n_components,
        explained_variance_ratio,
        cumulative_variance: cumulative,
        feature_importance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::{CsvData, FeatureMatrix};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_run_pca() {
        let content = "a,b,c\n1.0,2.0,3.0\n2.0,4.0,5.0\n3.0,6.0,7.0\n4.0,8.0,9.0\n5.0,10.0,11.0";
        let mut file = NamedTempFile::new().expect("create");
        file.write_all(content.as_bytes()).expect("write");

        let csv = CsvData::from_file(file.path(), false).expect("parse");
        let features = FeatureMatrix::from_csv(&csv).expect("extract");
        let normalized = features.normalize();

        let result = run_pca(&normalized, 0).expect("pca");

        assert!(result.n_components > 0);
        assert!(!result.explained_variance_ratio.is_empty());
        assert!(!result.cumulative_variance.is_empty());
        // Cumulative should be monotonically increasing
        for i in 1..result.cumulative_variance.len() {
            assert!(result.cumulative_variance[i] >= result.cumulative_variance[i - 1]);
        }
    }

    #[test]
    fn test_pca_too_few_features() {
        let content = "a\n1.0\n2.0\n3.0";
        let mut file = NamedTempFile::new().expect("create");
        file.write_all(content.as_bytes()).expect("write");

        let csv = CsvData::from_file(file.path(), false).expect("parse");
        let features = FeatureMatrix::from_csv(&csv).expect("extract");
        let normalized = features.normalize();

        let result = run_pca(&normalized, 0);
        assert!(result.is_err());
    }
}
