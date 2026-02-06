#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::uninlined_format_args)]

mod context;
mod csv_reader;
mod llm;
mod ml;
mod structs;
mod xml;

use clap::{Parser, Subcommand};
use structs::{CsvData, FeatureMatrix, Result, ZError};
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

        /// DBSCAN epsilon (0.0 = auto-estimate via k-distance heuristic)
        #[arg(long, default_value = "0.0")]
        dbscan_eps: f64,

        /// DBSCAN minimum points per cluster
        #[arg(long, default_value = "5")]
        dbscan_min_points: usize,

        /// Number of PCA components (0 = auto)
        #[arg(long, default_value = "0")]
        pca_components: usize,
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
            dbscan_eps,
            dbscan_min_points,
            pca_components,
        }) => run_analyze(
            &csv,
            &output_dir,
            &ml::pipeline::AnalysisConfig {
                clusters,
                dbscan_eps,
                dbscan_min_points,
                pca_components,
            },
            tsv,
        ),

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
fn run_analyze(
    csv_path: &Path,
    output_dir: &Path,
    config: &ml::pipeline::AnalysisConfig,
    tsv: bool,
) -> Result<()> {
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
    let csv_data = CsvData::from_file(csv_path, tsv)?;
    eprintln!(
        "Loaded {} rows x {} columns",
        csv_data.row_count(),
        csv_data.col_count()
    );

    // Extract and normalize features
    eprintln!("Extracting features...");
    let features = FeatureMatrix::from_csv(&csv_data)?;
    let normalized = features.normalize();

    // Run pipeline
    eprintln!("Running analysis pipeline...");
    let result = ml::pipeline::run_pipeline(&features, &normalized, config)?;

    // Write output files
    eprintln!("Writing output files...");

    let summary = ml::output::build_summary(csv_path, &csv_data, &result);
    ml::output::write_summary(output_dir, &summary)?;
    ml::output::write_clusters(output_dir, &result.cluster_result, &normalized)?;
    ml::output::write_anomalies(output_dir, &result.anomalies)?;

    let stats_refs: Vec<_> = result.column_stats.iter().collect();
    ml::output::write_stats_json(
        output_dir,
        &csv_data,
        &stats_refs,
        &result.cluster_result,
        &result.anomalies,
        result.dbscan_result.as_ref(),
        result.correlation.as_ref(),
        result.pca.as_ref(),
    )?;

    if let Some(corr) = &result.correlation {
        ml::output::write_correlation(output_dir, corr)?;
    }

    eprintln!("Output written to {}", output_dir.display());
    eprintln!("  - summary.txt");
    eprintln!("  - clusters.csv");
    eprintln!("  - anomalies.csv");
    eprintln!("  - stats.json");
    if result.correlation.is_some() {
        eprintln!("  - correlation.csv");
    }

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
