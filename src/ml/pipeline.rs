//! Analysis pipeline that orchestrates all ML computations

use crate::structs::{
    AnalysisResult, Anomaly, ColumnStats, FeatureMatrix, NormalizedFeatures, Result,
};

/// Configuration for the analysis pipeline
pub struct AnalysisConfig {
    pub clusters: usize,
    pub dbscan_eps: f64,
    pub dbscan_min_points: usize,
    pub pca_components: usize,
}

/// Run the full analysis pipeline
///
/// # Errors
/// Returns error if feature extraction or clustering fails.
/// DBSCAN, PCA, and correlation failures are non-fatal (logged and set to `None`).
#[allow(clippy::cast_precision_loss)]
pub fn run_pipeline(
    features: &FeatureMatrix,
    normalized: &NormalizedFeatures,
    config: &AnalysisConfig,
) -> Result<AnalysisResult> {
    // Column statistics
    let mut column_stats_with_data = Vec::new();
    for (i, name) in features.names.iter().enumerate() {
        if let Some(col) = features.column(i) {
            if let Ok(stats) = ColumnStats::calculate(name, &col) {
                column_stats_with_data.push((stats, col));
            }
        }
    }

    // K-means clustering
    let k = if config.clusters == 0 {
        super::clustering::suggest_k(normalized, 10)
    } else {
        config.clusters
    };
    let cluster_result = super::clustering::kmeans(normalized, k)?;

    // Anomaly detection (IQR outliers)
    let mut anomalies = Vec::new();
    for (stats, col) in &column_stats_with_data {
        let outlier_indices = stats.outlier_indices(col);
        for idx in outlier_indices {
            let value = col.get(idx).copied().unwrap_or(0.0);
            let z_score = if stats.std_dev > 0.0 {
                (value - stats.mean) / stats.std_dev
            } else {
                0.0
            };
            anomalies.push(Anomaly {
                row_id: idx,
                anomaly_type: format!("{}_outlier", stats.name),
                score: z_score.abs() / 4.0,
                details: format!(
                    "{}={:.2} is {:.1} std from mean",
                    stats.name, value, z_score
                ),
            });
        }
    }

    // DBSCAN (non-fatal)
    let dbscan_result = run_dbscan_safe(normalized, config, &mut anomalies);

    // Sort and dedupe anomalies
    anomalies.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen_rows = std::collections::HashSet::new();
    anomalies.retain(|a| seen_rows.insert(a.row_id));

    // Correlation (non-fatal)
    let correlation = match super::correlation::correlation_matrix(features) {
        Ok(corr) => Some(corr),
        Err(e) => {
            eprintln!("Warning: correlation failed: {e}");
            None
        }
    };

    // PCA (non-fatal)
    let pca = if features.n_features() >= 2 {
        match super::reduction::run_pca(normalized, config.pca_components) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("Warning: PCA failed: {e}");
                None
            }
        }
    } else {
        None
    };

    let column_stats = column_stats_with_data
        .into_iter()
        .map(|(s, _)| s)
        .collect();

    Ok(AnalysisResult {
        column_stats,
        cluster_result,
        dbscan_result,
        anomalies,
        correlation,
        pca,
    })
}

/// Run DBSCAN, adding noise points as anomalies. Non-fatal.
fn run_dbscan_safe(
    normalized: &NormalizedFeatures,
    config: &AnalysisConfig,
    anomalies: &mut Vec<Anomaly>,
) -> Option<crate::structs::DbscanResult> {
    let eps = if config.dbscan_eps <= 0.0 {
        super::clustering::estimate_epsilon(normalized, config.dbscan_min_points)
    } else {
        config.dbscan_eps
    };

    match super::clustering::dbscan(normalized, eps, config.dbscan_min_points) {
        Ok(result) => {
            // Add noise points as anomalies
            for (i, label) in result.labels.iter().enumerate() {
                if label.is_none() {
                    let row_id = normalized.row_indices[i];
                    anomalies.push(Anomaly {
                        row_id,
                        anomaly_type: "dbscan_noise".to_string(),
                        score: 0.8,
                        details: format!("Row {row_id} classified as noise by DBSCAN (eps={eps:.4})"),
                    });
                }
            }
            Some(result)
        }
        Err(e) => {
            eprintln!("Warning: DBSCAN failed: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::CsvData;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> CsvData {
        let content = "name,x,y,z\na,1.0,10.0,100.0\nb,2.0,20.0,200.0\nc,3.0,30.0,300.0\nd,4.0,40.0,400.0\ne,5.0,50.0,500.0\nf,100.0,1.0,1.0";
        let mut file = NamedTempFile::new().expect("create");
        file.write_all(content.as_bytes()).expect("write");
        CsvData::from_file(file.path(), false).expect("parse")
    }

    #[test]
    fn test_full_pipeline() {
        let csv = create_test_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract");
        let normalized = features.normalize();

        let config = AnalysisConfig {
            clusters: 2,
            dbscan_eps: 0.0,
            dbscan_min_points: 2,
            pca_components: 0,
        };

        let result = run_pipeline(&features, &normalized, &config).expect("pipeline");

        assert!(!result.column_stats.is_empty());
        assert_eq!(result.cluster_result.k, 2);
        assert!(result.correlation.is_some());
        assert!(result.pca.is_some());
    }

    #[test]
    fn test_pipeline_defaults() {
        let csv = create_test_csv();
        let features = FeatureMatrix::from_csv(&csv).expect("extract");
        let normalized = features.normalize();

        let config = AnalysisConfig {
            clusters: 0,
            dbscan_eps: 0.0,
            dbscan_min_points: 5,
            pca_components: 0,
        };

        let result = run_pipeline(&features, &normalized, &config).expect("pipeline");

        assert!(!result.column_stats.is_empty());
        assert!(!result.anomalies.is_empty());
    }
}
