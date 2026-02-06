use crate::structs::{ColumnStats, Result, ZError};

impl ColumnStats {
    /// Calculate statistics for a vector of values
    ///
    /// # Errors
    /// Returns error if values is empty
    #[allow(clippy::cast_precision_loss)]
    pub fn calculate(name: &str, values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Err(ZError::Ml("Cannot calculate stats for empty data".into()));
        }

        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[count - 1];

        let q1 = percentile(&sorted, 25.0);
        let median = percentile(&sorted, 50.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        Ok(Self {
            name: name.to_string(),
            count,
            mean,
            std_dev,
            min,
            max,
            q1,
            median,
            q3,
            iqr,
        })
    }
}

/// Calculate percentile using linear interpolation
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let k = (p / 100.0) * (sorted.len() - 1) as f64;
    let f = k.floor() as usize;
    let c = k.ceil() as usize;

    if f == c {
        sorted[f]
    } else {
        let d0 = sorted[f] * (c as f64 - k);
        let d1 = sorted[c] * (k - f as f64);
        d0 + d1
    }
}

/// Calculate correlation coefficient between two variables
///
/// # Errors
/// Returns error if vectors have different lengths or fewer than 2 values
#[allow(clippy::cast_precision_loss)]
pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(ZError::Ml("Vectors must have same length".into()));
    }
    if x.len() < 2 {
        return Err(ZError::Ml("Need at least 2 values for correlation".into()));
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }

    Ok(cov / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = ColumnStats::calculate("test", &values).expect("calculate stats");

        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 10.0).abs() < 0.01);
        assert!((stats.median - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_outlier_detection() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is outlier
        let stats = ColumnStats::calculate("test", &values).expect("calculate stats");
        let outliers = stats.outlier_indices(&values);

        assert!(outliers.contains(&5));
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y).expect("calculate correlation");

        assert!((corr - 1.0).abs() < 0.01); // Perfect positive correlation
    }
}
