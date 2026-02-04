use crate::error::{Result, ZError};
use crate::ml::features::NormalizedFeatures;
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array2;

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
    pub fn summary(&self) -> String {
        let mut s = format!("K-means clustering with k={}\n", self.k);
        for (i, size) in self.sizes.iter().enumerate() {
            s.push_str(&format!("  Cluster {}: {} samples\n", i, size));
        }
        s
    }

    /// Get row indices for a specific cluster
    #[allow(dead_code)]
    pub fn get_cluster(&self, cluster_id: usize) -> Option<&Vec<usize>> {
        self.cluster_members.get(cluster_id)
    }
}

/// Perform K-means clustering on normalized features
pub fn kmeans(features: &NormalizedFeatures, k: usize) -> Result<ClusterResult> {
    let n_samples = features.n_samples();

    if n_samples < k {
        return Err(ZError::Ml(format!(
            "Cannot create {} clusters with only {} samples",
            k, n_samples
        )));
    }

    if k == 0 {
        return Err(ZError::Ml("k must be at least 1".into()));
    }

    // Convert to ndarray Array2
    let flat_data: Vec<f64> = features.to_flat();
    let array = Array2::from_shape_vec((n_samples, features.n_features()), flat_data)
        .map_err(|e| ZError::Ml(format!("Failed to create array: {}", e)))?;

    // Create dataset
    let dataset = DatasetBase::from(array);

    // Run K-means
    let model = KMeans::params(k)
        .max_n_iterations(100)
        .tolerance(1e-4)
        .fit(&dataset)
        .map_err(|e| ZError::Ml(format!("K-means failed: {}", e)))?;

    let predictions = model.predict(&dataset);
    let labels: Vec<usize> = predictions.iter().copied().collect();

    // Calculate cluster sizes and members
    let mut sizes: Vec<usize> = vec![0usize; k];
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); k];

    for (sample_idx, &cluster_id) in labels.iter().enumerate() {
        sizes[cluster_id] += 1;
        // Map back to original row index
        let original_row = features.row_indices[sample_idx];
        cluster_members[cluster_id].push(original_row);
    }

    Ok(ClusterResult {
        labels,
        k,
        sizes,
        cluster_members,
    })
}

/// Find optimal k using elbow method (simplified)
/// Returns suggested k value based on diminishing returns
pub fn suggest_k(features: &NormalizedFeatures, max_k: usize) -> usize {
    let n = features.n_samples();
    let max_k = max_k.min(n).max(1);

    // Simple heuristic: sqrt of sample count, capped
    let suggested = (n as f64).sqrt().round() as usize;
    suggested.clamp(2, max_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csv_reader::CsvData;
    use crate::ml::features::FeatureMatrix;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_clusterable_csv() -> CsvData {
        // Create data with 2 clear clusters
        let content = r#"id,x,y
1,1.0,1.0
2,1.1,1.1
3,0.9,0.9
4,1.0,1.2
5,10.0,10.0
6,10.1,10.1
7,9.9,9.9
8,10.0,10.2"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        CsvData::from_file(file.path(), false).unwrap()
    }

    #[test]
    fn test_kmeans_clustering() {
        let csv = create_clusterable_csv();
        let features = FeatureMatrix::from_csv(&csv).unwrap();
        let normalized = features.normalize();

        let result = kmeans(&normalized, 2).unwrap();

        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 8);
        // Each cluster should have 4 members
        assert!(result.sizes.iter().all(|&s| s == 4));
    }

    #[test]
    fn test_suggest_k() {
        let csv = create_clusterable_csv();
        let features = FeatureMatrix::from_csv(&csv).unwrap();
        let normalized = features.normalize();

        let k = suggest_k(&normalized, 10);
        assert!(k >= 2 && k <= 10);
    }
}
