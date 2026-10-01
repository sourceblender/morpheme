use anyhow::Result;
use splinter::VERSION;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!(version = VERSION, "splinter-cli starting (stub)");
    println!("splinter {}", VERSION);
    println!("this binary is a stub — see docs/architecture.md");
    Ok(())
}