//! Context manager for lazy loading of ML output and instruction files

use crate::structs::{FileInfo, FileType, Result, ZError};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Size limits for 4GB VRAM constraint
pub const MAX_FILE_CONTENT: usize = 2000;
pub const MAX_CSV_ROWS: usize = 20;

/// Create file info from a path
fn file_info_from_path(path: &Path) -> Result<FileInfo> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ZError::Config("Invalid filename".into()))?
        .to_string();

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("txt");

    let file_type = FileType::from_extension(extension);

    let metadata = fs::metadata(path)?;
    let size_bytes = metadata.len();

    // Read first 100 chars for preview
    let content = fs::read_to_string(path)?;
    let preview = truncate_string(&content, 100);

    Ok(FileInfo {
        filename,
        file_type,
        size_bytes,
        preview,
    })
}

/// Manager for context files with lazy loading
pub struct ContextManager {
    context_dir: PathBuf,
    file_index: Vec<FileInfo>,
    // Lazy-loaded caches
    loaded_files: RefCell<HashMap<String, String>>,
}

impl ContextManager {
    /// Create a context manager from a directory
    ///
    /// # Errors
    /// Returns error if directory cannot be read
    pub fn from_directory(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            return Err(ZError::Config(format!(
                "Not a directory: {}",
                dir.display()
            )));
        }

        let mut file_index = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip directories and hidden files
            if path.is_dir() {
                continue;
            }
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('.'))
            {
                continue;
            }

            match file_info_from_path(&path) {
                Ok(info) => file_index.push(info),
                Err(e) => eprintln!("Warning: Could not index {}: {e}", path.display()),
            }
        }

        // Sort by filename for consistent ordering
        file_index.sort_by(|a, b| a.filename.cmp(&b.filename));

        Ok(ContextManager {
            context_dir: dir.to_path_buf(),
            file_index,
            loaded_files: RefCell::new(HashMap::new()),
        })
    }

    /// Get number of indexed files
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.file_index.len()
    }

    /// List all context files
    #[must_use]
    pub fn list_files(&self) -> &[FileInfo] {
        &self.file_index
    }

    /// Get file info by filename
    #[must_use]
    pub fn get_file_info(&self, filename: &str) -> Option<&FileInfo> {
        self.file_index.iter().find(|f| f.filename == filename)
    }

    /// Read a file's content (cached, truncated to `MAX_FILE_CONTENT`)
    ///
    /// # Errors
    /// Returns error if file not found or cannot be read
    pub fn read_file(&self, filename: &str) -> Result<String> {
        // Check cache
        if let Some(content) = self.loaded_files.borrow().get(filename) {
            return Ok(content.clone());
        }

        // Verify file exists in index
        if !self.file_index.iter().any(|f| f.filename == filename) {
            return Err(ZError::Config(format!("File not in context: {filename}")));
        }

        // Read file
        let path = self.context_dir.join(filename);
        let content = fs::read_to_string(&path)?;

        // Truncate if needed
        let truncated = if content.len() > MAX_FILE_CONTENT {
            format!(
                "{}...\n[Truncated: {} chars total]",
                truncate_string(&content, MAX_FILE_CONTENT),
                content.len()
            )
        } else {
            content.clone()
        };

        // Cache it
        self.loaded_files
            .borrow_mut()
            .insert(filename.to_string(), truncated.clone());

        Ok(truncated)
    }

    /// Query rows from a CSV file
    ///
    /// # Errors
    /// Returns error if file not found or not CSV
    pub fn query_csv(
        &self,
        filename: &str,
        filter: Option<&str>,
        limit: Option<usize>,
    ) -> Result<String> {
        let info = self
            .get_file_info(filename)
            .ok_or_else(|| ZError::Config(format!("File not found: {filename}")))?;

        if info.file_type != FileType::Csv {
            return Err(ZError::Config(format!("{filename} is not a CSV file")));
        }

        let path = self.context_dir.join(filename);
        let content = fs::read_to_string(&path)?;

        let limit = limit.unwrap_or(MAX_CSV_ROWS).min(MAX_CSV_ROWS);

        let mut lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Ok(String::new());
        }

        let header = lines.remove(0);

        // Apply filter if provided
        let filtered: Vec<&str> = if let Some(filter_str) = filter {
            lines
                .into_iter()
                .filter(|line| line.contains(filter_str))
                .take(limit)
                .collect()
        } else {
            lines.into_iter().take(limit).collect()
        };

        let mut result = String::from(header);
        result.push('\n');
        for line in filtered {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    /// Build file index summary for system prompt
    #[must_use]
    pub fn build_file_index_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut summary = String::new();
        for info in &self.file_index {
            let _ = writeln!(summary, "- {}", info.display());
        }
        summary
    }
}

/// Truncate a string to max chars, breaking at word boundary if possible
fn truncate_string(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }

    // Try to break at a word boundary
    let truncated = &s[..max_chars];
    if let Some(last_space) = truncated.rfind(char::is_whitespace) {
        if last_space > max_chars / 2 {
            return s[..last_space].to_string();
        }
    }

    truncated.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_context() -> TempDir {
        let dir = TempDir::new().expect("create temp dir");

        // Create summary.txt
        let mut f = fs::File::create(dir.path().join("summary.txt")).expect("create");
        writeln!(f, "ML Analysis Summary").expect("write");
        writeln!(f, "Rows: 100").expect("write");

        // Create clusters.csv
        let mut f = fs::File::create(dir.path().join("clusters.csv")).expect("create");
        writeln!(f, "row_id,cluster,distance").expect("write");
        writeln!(f, "0,0,0.1").expect("write");
        writeln!(f, "1,1,0.2").expect("write");
        writeln!(f, "2,0,0.15").expect("write");

        // Create instructions.txt
        let mut f = fs::File::create(dir.path().join("instructions.txt")).expect("create");
        writeln!(f, "Flag all anomalies in the XML").expect("write");

        dir
    }

    #[test]
    fn test_context_manager_creation() {
        let dir = create_test_context();
        let cm = ContextManager::from_directory(dir.path()).expect("create context manager");

        assert_eq!(cm.file_count(), 3);
    }

    #[test]
    fn test_list_files() {
        let dir = create_test_context();
        let cm = ContextManager::from_directory(dir.path()).expect("create context manager");

        let files = cm.list_files();
        let names: Vec<_> = files.iter().map(|f| f.filename.as_str()).collect();

        assert!(names.contains(&"summary.txt"));
        assert!(names.contains(&"clusters.csv"));
        assert!(names.contains(&"instructions.txt"));
    }

    #[test]
    fn test_read_file() {
        let dir = create_test_context();
        let cm = ContextManager::from_directory(dir.path()).expect("create context manager");

        let content = cm.read_file("summary.txt").expect("read file");
        assert!(content.contains("ML Analysis Summary"));
    }

    #[test]
    fn test_query_csv() {
        let dir = create_test_context();
        let cm = ContextManager::from_directory(dir.path()).expect("create context manager");

        let result = cm.query_csv("clusters.csv", None, Some(10)).expect("query");
        assert!(result.contains("row_id,cluster,distance"));
        assert!(result.contains("0,0,0.1"));

        // Test with filter - filter for rows containing ",0," (cluster 0)
        let filtered = cm
            .query_csv("clusters.csv", Some(",0,"), Some(10))
            .expect("query");
        assert!(filtered.contains("0,0,0.1"));
        assert!(filtered.contains("2,0,0.15"));
        // Should not contain row with cluster 1
        assert!(!filtered.contains("1,1,0.2"));
    }

    #[test]
    fn test_file_not_found() {
        let dir = create_test_context();
        let cm = ContextManager::from_directory(dir.path()).expect("create context manager");

        let result = cm.read_file("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_truncate_string() {
        let long = "This is a very long string that should be truncated";
        let truncated = truncate_string(long, 20);
        assert!(truncated.len() <= 20);

        let short = "Short";
        let not_truncated = truncate_string(short, 20);
        assert_eq!(not_truncated, "Short");
    }
}
