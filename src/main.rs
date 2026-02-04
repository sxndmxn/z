#![allow(clippy::uninlined_format_args)]

use std::fmt::Write as _;

mod csv_reader;
mod db;
mod error;
mod llm;
mod ml;
mod xml;

use clap::Parser;
use db::DataSource;
use error::{Result, ZError};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Z - LLM tool for XML modification with ML analysis
#[derive(Parser, Debug)]
#[command(name = "z")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CSV/TSV file to analyze
    #[arg(short, long)]
    input: PathBuf,

    /// XML file to modify
    #[arg(short = 'x', long)]
    xml: PathBuf,

    /// Path to llama-server executable
    #[arg(short, long)]
    server: PathBuf,

    /// Path to GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to database file (JSON)
    #[arg(short, long)]
    database: PathBuf,

    /// Number of clusters for K-means (0 = auto)
    #[arg(short = 'k', long, default_value = "0")]
    clusters: usize,

    /// Treat input as TSV instead of CSV
    #[arg(long)]
    tsv: bool,

    /// Parent element in XML to insert rows into
    #[arg(long, default_value = "items")]
    parent_element: String,

    /// Element name for inserted rows
    #[arg(long, default_value = "item")]
    element_name: String,

    /// Context size for LLM (tokens)
    #[arg(long, default_value = "12000")]
    context_size: u32,

    /// GPU layers to offload
    #[arg(long, default_value = "99")]
    gpu_layers: u32,

    /// Maximum conversation turns
    #[arg(long, default_value = "10")]
    max_turns: usize,

    /// Dry run - don't modify XML, just show what would be selected
    #[arg(long)]
    dry_run: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn run() -> Result<()> {
    let args = Args::parse();

    // Validate paths
    if !args.input.exists() {
        return Err(ZError::Config(format!(
            "Input file not found: {}",
            args.input.display()
        )));
    }
    if !args.xml.exists() {
        return Err(ZError::Config(format!(
            "XML file not found: {}",
            args.xml.display()
        )));
    }
    if !args.server.exists() {
        return Err(ZError::Config(format!(
            "Server executable not found: {}",
            args.server.display()
        )));
    }
    if !args.model.exists() {
        return Err(ZError::Config(format!(
            "Model file not found: {}",
            args.model.display()
        )));
    }
    if !args.database.exists() {
        return Err(ZError::Config(format!(
            "Database file not found: {}",
            args.database.display()
        )));
    }

    // Setup shutdown flag for cleanup
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    // Setup Ctrl+C handler
    ctrlc::set_handler(move || {
        eprintln!("\nReceived Ctrl+C, shutting down...");
        shutdown_clone.store(true, Ordering::SeqCst);
    })
    .map_err(|e| ZError::Config(format!("Failed to set Ctrl+C handler: {e}")))?;

    // Setup panic hook
    llm::server::setup_panic_hook(shutdown.clone());

    // Phase 1: Parse CSV
    eprintln!("Reading input file: {}", args.input.display());
    let csv_data = csv_reader::CsvData::from_file(&args.input, args.tsv)?;
    eprintln!(
        "Loaded {} rows x {} columns",
        csv_data.row_count(),
        csv_data.col_count()
    );

    // Phase 2: ML Analysis
    eprintln!("Extracting features...");
    let features = ml::features::FeatureMatrix::from_csv(&csv_data)?;
    let normalized = features.normalize();

    eprintln!("Computing statistics...");
    let mut stats_summary = String::new();
    for (i, name) in features.names.iter().enumerate() {
        if let Some(col) = features.column(i) {
            if let Ok(stats) = ml::stats::ColumnStats::calculate(name, &col) {
                stats_summary.push_str(&stats.summary());
                stats_summary.push('\n');

                let outliers = stats.outlier_indices(&col);
                if !outliers.is_empty() {
                    let _ = writeln!(stats_summary, "  Outliers at indices: {outliers:?}");
                }
            }
        }
    }

    // Clustering
    let k = if args.clusters == 0 {
        ml::clustering::suggest_k(&normalized, 10)
    } else {
        args.clusters
    };
    eprintln!("Running K-means with k={k}...");

    let cluster_result = ml::clustering::kmeans(&normalized, k)?;
    let cluster_summary = cluster_result.summary();

    // Build ML summary
    let ml_summary = format!("{stats_summary}\n{cluster_summary}");
    let csv_summary = csv_data.summary();

    // Phase 3: Load database
    eprintln!("Loading database: {}", args.database.display());
    let data_source = db::JsonDataSource::from_file(&args.database)?;
    let available_ids = data_source.get_all_ids()?;
    eprintln!("Database contains {} rows", available_ids.len());

    // Check for shutdown before expensive LLM startup
    if shutdown.load(Ordering::SeqCst) {
        eprintln!("Shutdown requested, exiting early");
        return Ok(());
    }

    // Phase 4: Start LLM server
    eprintln!("Starting LLM server...");
    let server_path = args.server.to_str()
        .ok_or_else(|| ZError::Config("Server path contains invalid UTF-8".into()))?;
    let model_path = args.model.to_str()
        .ok_or_else(|| ZError::Config("Model path contains invalid UTF-8".into()))?;

    let server = llm::LlamaServer::spawn(
        server_path,
        model_path,
        args.context_size,
        args.gpu_layers,
    )?;

    // Check for shutdown after server start
    if shutdown.load(Ordering::SeqCst) {
        eprintln!("Shutdown requested, stopping server");
        return Ok(());
    }

    // Phase 5: Run LLM conversation
    let system_prompt = llm::build_system_prompt(&ml_summary, &csv_summary);
    let mut client = llm::LlmClient::new(&server, &system_prompt, args.max_turns);

    // Initial prompt to start the conversation
    client.add_user_message(
        "Please analyze the data and select appropriate rows to add to the XML file.",
    );

    let selected_rows = client.run_conversation(&data_source)?;

    // Report token usage
    let usage = client.total_usage();
    eprintln!(
        "Token usage: {} prompt + {} completion = {} total",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    );

    if selected_rows.is_empty() {
        eprintln!("No rows were selected by the LLM");
        return Ok(());
    }

    eprintln!("Selected {} rows: {selected_rows:?}", selected_rows.len());

    // Phase 6: Modify XML
    if args.dry_run {
        eprintln!("Dry run - not modifying XML");
        println!("Would insert rows: {selected_rows:?}");
        return Ok(());
    }

    eprintln!("Loading XML file: {}", args.xml.display());
    let xml_modifier = xml::XmlModifier::from_file(&args.xml)?;

    // Get full row data for selected IDs
    let mut rows_to_insert = Vec::new();
    for id in &selected_rows {
        if let Some(row) = data_source.get_row(id)? {
            rows_to_insert.push(row);
        }
    }

    eprintln!("Inserting {} rows into XML...", rows_to_insert.len());
    let modified_xml =
        xml_modifier.insert_rows(&rows_to_insert, &args.parent_element, &args.element_name)?;

    xml::XmlModifier::write_to_file(&modified_xml, &args.xml)?;
    eprintln!("XML file updated: {}", args.xml.display());

    Ok(())
}
