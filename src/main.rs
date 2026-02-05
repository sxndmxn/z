#![allow(clippy::uninlined_format_args)]

mod context;
mod csv_reader;
mod db;
mod error;
mod llm;
mod ml;
mod xml;

use clap::{Parser, Subcommand};
use error::{Result, ZError};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Z - LLM tool for XML modification with ML analysis
#[derive(Parser, Debug)]
#[command(name = "z")]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run ML analysis on CSV, output summary files
    Analyze {
        /// Input CSV/TSV file to analyze
        #[arg(short, long)]
        csv: PathBuf,

        /// Output directory for ML results
        #[arg(short, long, default_value = "./ml_output")]
        output_dir: PathBuf,

        /// Number of clusters for K-means (0 = auto)
        #[arg(short = 'k', long, default_value = "0")]
        clusters: usize,

        /// Treat input as TSV instead of CSV
        #[arg(long)]
        tsv: bool,
    },

    /// Use LLM to modify XML based on context files
    Modify {
        /// Directory containing context files (ML outputs, instructions)
        #[arg(short, long)]
        context_dir: PathBuf,

        /// XML file to modify
        #[arg(short = 'x', long)]
        xml: PathBuf,

        /// Path to llama-server executable
        #[arg(short, long)]
        server: PathBuf,

        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Context size for LLM (tokens)
        #[arg(long, default_value = "12000")]
        context_size: u32,

        /// GPU layers to offload
        #[arg(long, default_value = "99")]
        gpu_layers: u32,

        /// Maximum conversation turns
        #[arg(long, default_value = "10")]
        max_turns: usize,

        /// Dry run - don't modify XML, just show what would be done
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Some(Commands::Analyze {
            csv,
            output_dir,
            clusters,
            tsv,
        }) => run_analyze(&csv, &output_dir, clusters, tsv),

        Some(Commands::Modify {
            context_dir,
            xml,
            server,
            model,
            context_size,
            gpu_layers,
            max_turns,
            dry_run,
        }) => run_modify(
            &context_dir,
            &xml,
            &server,
            &model,
            context_size,
            gpu_layers,
            max_turns,
            dry_run,
        ),

        None => {
            eprintln!("No subcommand provided. Use 'z analyze' or 'z modify'.");
            eprintln!("Run 'z --help' for usage information.");
            std::process::exit(1);
        }
    }
}

/// Run the ML analysis phase
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn run_analyze(csv_path: &Path, output_dir: &Path, clusters: usize, tsv: bool) -> Result<()> {
    use std::fmt::Write as _;

    // Validate input
    if !csv_path.exists() {
        return Err(ZError::Config(format!(
            "CSV file not found: {}",
            csv_path.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    eprintln!("Analyzing: {}", csv_path.display());

    // Parse CSV
    let csv_data = csv_reader::CsvData::from_file(csv_path, tsv)?;
    eprintln!(
        "Loaded {} rows x {} columns",
        csv_data.row_count(),
        csv_data.col_count()
    );

    // Extract features
    eprintln!("Extracting features...");
    let features = ml::features::FeatureMatrix::from_csv(&csv_data)?;
    let normalized = features.normalize();

    // Compute statistics
    eprintln!("Computing statistics...");
    let mut column_stats = Vec::new();
    for (i, name) in features.names.iter().enumerate() {
        if let Some(col) = features.column(i) {
            if let Ok(stats) = ml::stats::ColumnStats::calculate(name, &col) {
                column_stats.push((stats, col));
            }
        }
    }

    // Clustering
    let k = if clusters == 0 {
        ml::clustering::suggest_k(&normalized, 10)
    } else {
        clusters
    };
    eprintln!("Running K-means with k={k}...");
    let cluster_result = ml::clustering::kmeans(&normalized, k)?;

    // Detect anomalies
    eprintln!("Detecting anomalies...");
    let mut anomalies = Vec::new();
    for (stats, col) in &column_stats {
        let outlier_indices = stats.outlier_indices(col);
        for idx in outlier_indices {
            let value = col.get(idx).copied().unwrap_or(0.0);
            let z_score = if stats.std_dev > 0.0 {
                (value - stats.mean) / stats.std_dev
            } else {
                0.0
            };
            anomalies.push(ml::output::Anomaly {
                row_id: idx,
                anomaly_type: format!("{}_outlier", stats.name),
                score: z_score.abs() / 4.0, // Normalize to ~0-1 range
                details: format!(
                    "{}={:.2} is {:.1} std from mean",
                    stats.name, value, z_score
                ),
            });
        }
    }

    // Sort anomalies by score (highest first) and dedupe by row_id
    anomalies.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen_rows = std::collections::HashSet::new();
    anomalies.retain(|a| seen_rows.insert(a.row_id));

    // Write output files
    eprintln!("Writing output files...");

    // Build summary
    let mut summary = String::new();
    let _ = writeln!(
        summary,
        "ML Analysis Summary for {}",
        csv_path
            .file_name()
            .map_or("input", |n| n.to_str().unwrap_or("input"))
    );
    let _ = writeln!(summary, "================================");
    let _ = writeln!(
        summary,
        "Rows: {}\nColumns: {} ({} numeric)",
        csv_data.row_count(),
        csv_data.col_count(),
        features.names.len()
    );
    let _ = writeln!(summary);
    let _ = writeln!(summary, "Key Statistics:");
    for (stats, _) in &column_stats {
        let _ = writeln!(summary, "- {}", stats.summary());
    }
    let _ = writeln!(summary);
    let _ = writeln!(summary, "Clustering (k={}):", cluster_result.k);
    for (i, size) in cluster_result.sizes.iter().enumerate() {
        let pct = (*size as f64 / csv_data.row_count() as f64) * 100.0;
        let _ = writeln!(summary, "- Cluster {i} ({pct:.0}%): {size} samples");
    }
    let _ = writeln!(summary);
    let _ = writeln!(summary, "Anomalies Detected: {} rows", anomalies.len());

    ml::output::write_summary(output_dir, &summary)?;

    // Write clusters.csv
    ml::output::write_clusters(output_dir, &cluster_result, &normalized)?;

    // Write anomalies.csv
    ml::output::write_anomalies(output_dir, &anomalies)?;

    // Write stats.json
    let stats_only: Vec<_> = column_stats.iter().map(|(s, _)| s).collect();
    ml::output::write_stats_json(
        output_dir,
        &csv_data,
        &stats_only,
        &cluster_result,
        &anomalies,
    )?;

    eprintln!("Output written to {}", output_dir.display());
    eprintln!("  - summary.txt");
    eprintln!("  - clusters.csv");
    eprintln!("  - anomalies.csv");
    eprintln!("  - stats.json");

    Ok(())
}

/// Run the LLM modification phase
#[allow(clippy::too_many_arguments)]
fn run_modify(
    context_dir: &Path,
    xml_path: &Path,
    server_path: &Path,
    model_path: &Path,
    context_size: u32,
    gpu_layers: u32,
    max_turns: usize,
    dry_run: bool,
) -> Result<()> {
    // Validate paths
    if !context_dir.exists() {
        return Err(ZError::Config(format!(
            "Context directory not found: {}",
            context_dir.display()
        )));
    }
    if !xml_path.exists() {
        return Err(ZError::Config(format!(
            "XML file not found: {}",
            xml_path.display()
        )));
    }
    if !server_path.exists() {
        return Err(ZError::Config(format!(
            "Server executable not found: {}",
            server_path.display()
        )));
    }
    if !model_path.exists() {
        return Err(ZError::Config(format!(
            "Model file not found: {}",
            model_path.display()
        )));
    }

    // Setup shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    ctrlc::set_handler(move || {
        eprintln!("\nReceived Ctrl+C, shutting down...");
        shutdown_clone.store(true, Ordering::SeqCst);
    })
    .map_err(|e| ZError::Config(format!("Failed to set Ctrl+C handler: {e}")))?;

    llm::server::setup_panic_hook(shutdown.clone());

    // Load context
    eprintln!("Loading context from: {}", context_dir.display());
    let context_manager = context::ContextManager::from_directory(context_dir)?;
    eprintln!("Found {} context files", context_manager.file_count());

    // Load XML
    eprintln!("Loading XML: {}", xml_path.display());
    let xml_modifier = xml::XmlModifier::from_file(xml_path)?;

    // Check for shutdown before LLM startup
    if shutdown.load(Ordering::SeqCst) {
        eprintln!("Shutdown requested, exiting early");
        return Ok(());
    }

    // Start LLM server
    eprintln!("Starting LLM server...");
    let server_str = server_path
        .to_str()
        .ok_or_else(|| ZError::Config("Server path contains invalid UTF-8".into()))?;
    let model_str = model_path
        .to_str()
        .ok_or_else(|| ZError::Config("Model path contains invalid UTF-8".into()))?;

    let server = llm::LlamaServer::spawn(server_str, model_str, context_size, gpu_layers)?;

    if shutdown.load(Ordering::SeqCst) {
        eprintln!("Shutdown requested, stopping server");
        return Ok(());
    }

    // Build system prompt
    let system_prompt = llm::build_modify_system_prompt(&context_manager);

    // Run conversation
    let mut client = llm::LlmClient::new(&server, &system_prompt, max_turns);
    client.add_user_message(
        "Please read the context files to understand the ML analysis, then modify the XML file accordingly.",
    );

    let modifications = client.run_modify_conversation(&context_manager, &xml_modifier)?;

    // Report usage
    let usage = client.total_usage();
    eprintln!(
        "Token usage: {} prompt + {} completion = {} total",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    );

    if modifications.is_empty() {
        eprintln!("No modifications were made");
        return Ok(());
    }

    eprintln!("Applied {} modifications", modifications.len());

    if dry_run {
        eprintln!("Dry run - not saving XML");
        for (i, m) in modifications.iter().enumerate() {
            eprintln!("  {}: {m}", i + 1);
        }
        return Ok(());
    }

    // Get modified XML and write
    let modified_xml = xml_modifier.get_content();
    xml::XmlModifier::write_to_file(&modified_xml, xml_path)?;
    eprintln!("XML updated: {}", xml_path.display());

    Ok(())
}
