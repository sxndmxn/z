//! ML output file writers for the analyze phase

use crate::csv_reader::CsvData;
use crate::error::Result;
use crate::ml::clustering::ClusterResult;
use crate::ml::features::NormalizedFeatures;
use crate::ml::stats::ColumnStats;
use serde::Serialize;
use std::fs;
use std::path::Path;

/// Represents an anomaly detected in the data
#[derive(Debug, Clone, Serialize)]
#[allow(clippy::struct_field_names)]
pub struct Anomaly {
    pub row_id: usize,
    pub anomaly_type: String,
    pub score: f64,
    pub details: String,
}

/// Write `summary.txt` - human/LLM readable overview
///
/// # Errors
/// Returns error if file cannot be written
pub fn write_summary(output_dir: &Path, content: &str) -> Result<()> {
    let path = output_dir.join("summary.txt");
    fs::write(path, content)?;
    Ok(())
}

/// Write `clusters.csv` - cluster assignments for each row
///
/// # Errors
/// Returns error if file cannot be written
#[allow(clippy::cast_precision_loss)]
pub fn write_clusters(
    output_dir: &Path,
    clusters: &ClusterResult,
    features: &NormalizedFeatures,
) -> Result<()> {
    use std::fmt::Write as _;

    let path = output_dir.join("clusters.csv");
    let mut content = String::from("row_id,cluster,distance_to_centroid\n");

    // Calculate centroids
    let mut centroids: Vec<Vec<f64>> = vec![vec![0.0; features.n_features()]; clusters.k];
    let mut counts = vec![0usize; clusters.k];

    for (sample_idx, &cluster_id) in clusters.labels.iter().enumerate() {
        counts[cluster_id] += 1;
        for (feat_idx, &val) in features.data[sample_idx].iter().enumerate() {
            centroids[cluster_id][feat_idx] += val;
        }
    }

    // Average centroids
    for (cluster_id, centroid) in centroids.iter_mut().enumerate() {
        if counts[cluster_id] > 0 {
            for val in centroid.iter_mut() {
                *val /= counts[cluster_id] as f64;
            }
        }
    }

    // Write rows with distances
    for (sample_idx, &cluster_id) in clusters.labels.iter().enumerate() {
        let original_row = features.row_indices[sample_idx];
        let distance = euclidean_distance(&features.data[sample_idx], &centroids[cluster_id]);
        let _ = writeln!(content, "{original_row},{cluster_id},{distance:.4}");
    }

    fs::write(path, content)?;
    Ok(())
}

/// Write `anomalies.csv` - detected anomalies
///
/// # Errors
/// Returns error if file cannot be written
pub fn write_anomalies(output_dir: &Path, anomalies: &[Anomaly]) -> Result<()> {
    use std::fmt::Write as _;

    let path = output_dir.join("anomalies.csv");
    let mut content = String::from("row_id,anomaly_type,score,details\n");

    for anomaly in anomalies {
        // Escape details for CSV
        let escaped_details = anomaly.details.replace('"', "\"\"");
        let _ = writeln!(
            content,
            "{},{},{:.4},\"{escaped_details}\"",
            anomaly.row_id, anomaly.anomaly_type, anomaly.score
        );
    }

    fs::write(path, content)?;
    Ok(())
}

/// Write `stats.json` - machine-readable statistics
///
/// # Errors
/// Returns error if file cannot be written
#[allow(clippy::cast_precision_loss)]
pub fn write_stats_json(
    output_dir: &Path,
    csv_data: &CsvData,
    stats: &[&ColumnStats],
    clusters: &ClusterResult,
    anomalies: &[Anomaly],
) -> Result<()> {
    let path = output_dir.join("stats.json");

    let stats_json: Vec<_> = stats
        .iter()
        .map(|s| StatsEntry {
            name: s.name.clone(),
            count: s.count,
            mean: s.mean,
            std_dev: s.std_dev,
            min: s.min,
            max: s.max,
            q1: s.q1,
            median: s.median,
            q3: s.q3,
            iqr: s.iqr,
        })
        .collect();

    let cluster_summary: Vec<_> = clusters
        .sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| ClusterEntry {
            id: i,
            size,
            percentage: (size as f64 / csv_data.row_count() as f64) * 100.0,
        })
        .collect();

    let output = StatsOutput {
        row_count: csv_data.row_count(),
        column_count: csv_data.col_count(),
        columns: csv_data.headers.clone(),
        statistics: stats_json,
        clustering: ClusteringSummary {
            k: clusters.k,
            clusters: cluster_summary,
        },
        anomalies_summary: AnomaliesSummary {
            total: anomalies.len(),
            by_type: count_by_type(anomalies),
        },
    };

    let json = serde_json::to_string_pretty(&output)?;
    fs::write(path, json)?;
    Ok(())
}

/// Calculate Euclidean distance between two points
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Count anomalies by type
fn count_by_type(anomalies: &[Anomaly]) -> Vec<AnomalyTypeCount> {
    use std::collections::HashMap;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for a in anomalies {
        *counts.entry(&a.anomaly_type).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .map(|(t, c)| AnomalyTypeCount {
            anomaly_type: t.to_string(),
            count: c,
        })
        .collect()
}

// JSON output structures

#[derive(Serialize)]
struct StatsOutput {
    row_count: usize,
    column_count: usize,
    columns: Vec<String>,
    statistics: Vec<StatsEntry>,
    clustering: ClusteringSummary,
    anomalies_summary: AnomaliesSummary,
}

#[derive(Serialize)]
struct StatsEntry {
    name: String,
    count: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    q1: f64,
    median: f64,
    q3: f64,
    iqr: f64,
}

#[derive(Serialize)]
struct ClusteringSummary {
    k: usize,
    clusters: Vec<ClusterEntry>,
}

#[derive(Serialize)]
struct ClusterEntry {
    id: usize,
    size: usize,
    percentage: f64,
}

#[derive(Serialize)]
struct AnomaliesSummary {
    total: usize,
    by_type: Vec<AnomalyTypeCount>,
}

#[derive(Serialize)]
struct AnomalyTypeCount {
    anomaly_type: String,
    count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_write_summary() {
        let dir = TempDir::new().expect("create temp dir");
        write_summary(dir.path(), "Test summary content").expect("write summary");

        let content = fs::read_to_string(dir.path().join("summary.txt")).expect("read");
        assert_eq!(content, "Test summary content");
    }

    #[test]
    fn test_write_anomalies() {
        let dir = TempDir::new().expect("create temp dir");
        let anomalies = vec![
            Anomaly {
                row_id: 1,
                anomaly_type: "price_outlier".to_string(),
                score: 0.95,
                details: "price=999 is 4.2 std above mean".to_string(),
            },
            Anomaly {
                row_id: 5,
                anomaly_type: "rating_outlier".to_string(),
                score: 0.87,
                details: "rating=1.0 with price=150+".to_string(),
            },
        ];

        write_anomalies(dir.path(), &anomalies).expect("write anomalies");

        let content = fs::read_to_string(dir.path().join("anomalies.csv")).expect("read");
        assert!(content.contains("row_id,anomaly_type,score,details"));
        assert!(content.contains("1,price_outlier,0.9500"));
        assert!(content.contains("5,rating_outlier,0.8700"));
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }
}
