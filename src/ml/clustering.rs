use crate::structs::{ClusterResult, DbscanResult, NormalizedFeatures, Result, ZError};
use linfa::traits::{Fit, Predict, Transformer};
use linfa::ParamGuard;
use linfa::DatasetBase;
use linfa_clustering::{Dbscan, KMeans};
use ndarray::Array2;

/// Perform K-means clustering on normalized features
///
/// # Errors
/// Returns error if clustering fails
pub fn kmeans(features: &NormalizedFeatures, k: usize) -> Result<ClusterResult> {
    let n_samples = features.n_samples();

    if n_samples < k {
        return Err(ZError::Ml(format!(
            "Cannot create {k} clusters with only {n_samples} samples"
        )));
    }

    if k == 0 {
        return Err(ZError::Ml("k must be at least 1".into()));
    }

    // Convert to ndarray Array2
    let flat_data: Vec<f64> = features.to_flat();
    let array = Array2::from_shape_vec((n_samples, features.n_features()), flat_data)
        .map_err(|e| ZError::Ml(format!("Failed to create array: {e}")))?;

    // Create dataset
    let dataset = DatasetBase::from(array);

    // Run K-means
    let model = KMeans::params(k)
        .max_n_iterations(100)
        .tolerance(1e-4)
        .fit(&dataset)
        .map_err(|e| ZError::Ml(format!("K-means failed: {e}")))?;

    let predictions = model.predict(&dataset);
    let labels: Vec<usize> = predictions.iter().copied().collect();

    // Calculate cluster sizes
    let mut sizes: Vec<usize> = vec![0usize; k];

    for &cluster_id in &labels {
        sizes[cluster_id] += 1;
    }

    Ok(ClusterResult {
        labels,
        k,
        sizes,
    })
}

/// Find optimal k using elbow method (simplified)
/// Returns suggested k value based on diminishing returns
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn suggest_k(features: &NormalizedFeatures, max_k: usize) -> usize {
    let n = features.n_samples();
    let max_k = max_k.min(n).max(1);

    // Simple heuristic: sqrt of sample count, capped
    let suggested = (n as f64).sqrt().round() as usize;
    suggested.clamp(2, max_k)
}

/// Estimate a good epsilon for DBSCAN using k-distance heuristic
///
/// Computes the k-th nearest neighbor distance for each point,
/// sorts them, and picks the "knee" (point of max curvature).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_epsilon(features: &NormalizedFeatures, min_points: usize) -> f64 {
    let n = features.n_samples();
    if n < min_points + 1 {
        return 0.5;
    }

    // Compute k-th nearest neighbor distance for each point
    let mut k_distances: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let mut distances: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                features.data[i]
                    .iter()
                    .zip(features.data[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // k-th nearest neighbor (0-indexed, so min_points - 1)
        let k_idx = (min_points - 1).min(distances.len() - 1);
        k_distances.push(distances[k_idx]);
    }

    // Sort k-distances
    k_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Find the "knee" - point of maximum second derivative
    if k_distances.len() < 3 {
        return k_distances.last().copied().unwrap_or(0.5);
    }

    let mut max_diff = 0.0f64;
    let mut knee_idx = k_distances.len() * 9 / 10; // default to 90th percentile

    for i in 1..k_distances.len() - 1 {
        let second_deriv = (k_distances[i + 1] - k_distances[i])
            - (k_distances[i] - k_distances[i - 1]);
        if second_deriv > max_diff {
            max_diff = second_deriv;
            knee_idx = i;
        }
    }

    k_distances[knee_idx]
}

/// Run DBSCAN clustering on normalized features
///
/// # Errors
/// Returns error if clustering fails
pub fn dbscan(
    features: &NormalizedFeatures,
    epsilon: f64,
    min_points: usize,
) -> Result<DbscanResult> {
    let n_samples = features.n_samples();
    let n_features = features.n_features();

    if n_samples < min_points {
        return Err(ZError::Ml(format!(
            "Need at least {min_points} samples for DBSCAN, got {n_samples}"
        )));
    }

    let flat_data = features.to_flat();
    let array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| ZError::Ml(format!("Failed to create array for DBSCAN: {e}")))?;

    let params = Dbscan::params(min_points)
        .tolerance(epsilon)
        .check()
        .map_err(|e| ZError::Ml(format!("DBSCAN params invalid: {e}")))?;

    let clusters = params.transform(&array);

    let labels: Vec<Option<usize>> = clusters.iter().copied().collect();

    // Count clusters and noise
    let mut n_clusters = 0usize;
    let mut n_noise = 0usize;
    let mut cluster_sizes = std::collections::HashMap::new();

    for label in &labels {
        match label {
            Some(c) => {
                *cluster_sizes.entry(*c).or_insert(0usize) += 1;
                if *c >= n_clusters {
                    n_clusters = *c + 1;
                }
            }
            None => n_noise += 1,
        }
    }

    let sizes: Vec<usize> = (0..n_clusters)
        .map(|c| cluster_sizes.get(&c).copied().unwrap_or(0))
        .collect();

    Ok(DbscanResult {
        labels,
        n_clusters,
        n_noise,
        sizes,
        epsilon,
        min_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::{CsvData, FeatureMatrix};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_clusterable_csv() -> CsvData {
        // Create data with 2 clear clusters
        let content = r"id,x,y
1,1.0,1.0
2,1.1,1.1
3,0.9,0.9
4,1.0,1.2
5,10.0,10.0
6,10.1,10.1
7,9.9,9.9
8,10.0,10.2";
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(content.as_bytes()).expect("write content");
        CsvData::from_file(file.path(), false).expect("parse csv")
    }

    #[test]
    fn test_kmeans_clustering() {
        let csv = create_clusterable_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract features");
        let normalized = features.normalize();

        let result = kmeans(&normalized, 2).expect("run kmeans");

        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 8);
        // Each cluster should have 4 members
        assert!(result.sizes.iter().all(|&s| s == 4));
    }

    #[test]
    fn test_suggest_k() {
        let csv = create_clusterable_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract features");
        let normalized = features.normalize();

        let k = suggest_k(&normalized, 10);
        assert!(k >= 2 && k <= 10);
    }

    #[test]
    fn test_dbscan() {
        let csv = create_clusterable_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract features");
        let normalized = features.normalize();

        let eps = estimate_epsilon(&normalized, 3);
        assert!(eps > 0.0);

        let result = dbscan(&normalized, eps, 3).expect("dbscan");
        assert_eq!(result.labels.len(), 8);
        assert!(result.n_clusters > 0);
    }

    #[test]
    fn test_estimate_epsilon() {
        let csv = create_clusterable_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract features");
        let normalized = features.normalize();

        let eps = estimate_epsilon(&normalized, 3);
        assert!(eps > 0.0);
        assert!(eps < 10.0);
    }
}
