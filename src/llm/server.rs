use crate::error::{Result, ZError};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Manages the llama-server child process
pub struct LlamaServer {
    child: Option<Child>,
    port: u16,
    #[allow(dead_code)]
    shutdown: Arc<AtomicBool>,
}

impl LlamaServer {
    /// Find an available port by binding to port 0
    fn find_available_port() -> Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| ZError::LlmServer(format!("Failed to find available port: {e}")))?;
        let port = listener.local_addr()
            .map_err(|e| ZError::LlmServer(format!("Failed to get port: {e}")))?
            .port();
        Ok(port)
    }

    /// Spawn llama-server with the given model
    ///
    /// # Errors
    /// Returns error if server fails to start
    pub fn spawn(server_path: &str, model_path: &str, context_size: u32, gpu_layers: u32) -> Result<Self> {
        let port = Self::find_available_port()?;
        let shutdown = Arc::new(AtomicBool::new(false));

        eprintln!("Starting llama-server on port {port}...");

        let child = Command::new(server_path)
            .args([
                "-m", model_path,
                "--port", &port.to_string(),
                "-c", &context_size.to_string(),
                "-ngl", &gpu_layers.to_string(),
                "--log-disable",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ZError::LlmServer(format!("Failed to spawn llama-server: {e}")))?;

        let server = LlamaServer {
            child: Some(child),
            port,
            shutdown,
        };

        // Wait for server to be ready
        server.wait_for_health(Duration::from_secs(30))?;

        eprintln!("llama-server ready on port {port}");
        Ok(server)
    }

    /// Get the server URL
    #[must_use]
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Get the completions endpoint
    #[must_use]
    pub fn completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.url())
    }

    /// Poll health endpoint until ready or timeout
    fn wait_for_health(&self, timeout: Duration) -> Result<()> {
        let health_url = format!("{}/health", self.url());
        let start = Instant::now();
        let poll_interval = Duration::from_millis(500);

        loop {
            if start.elapsed() > timeout {
                return Err(ZError::LlmServer(format!(
                    "Server failed to start within {timeout:?}"
                )));
            }

            match ureq::get(&health_url).timeout(Duration::from_secs(2)).call() {
                Ok(response) if response.status() == 200 => {
                    return Ok(());
                }
                _ => {
                    thread::sleep(poll_interval);
                }
            }
        }
    }

    /// Check if server is healthy
    #[allow(dead_code)]
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        let health_url = format!("{}/health", self.url());
        matches!(
            ureq::get(&health_url).timeout(Duration::from_secs(2)).call(),
            Ok(response) if response.status() == 200
        )
    }

    /// Signal shutdown
    #[allow(dead_code)]
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Kill the server process
    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            eprintln!("Stopping llama-server...");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Setup panic hook to kill server on panic
pub fn setup_panic_hook(shutdown_flag: Arc<AtomicBool>) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        shutdown_flag.store(true, Ordering::SeqCst);
        default_hook(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_available_port() {
        let port = LlamaServer::find_available_port().expect("find port");
        assert!(port > 0);
    }
}
