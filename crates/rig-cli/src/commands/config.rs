use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use rig_core::RigConfig;

use crate::config::RigConfigLoader;

#[derive(Debug, Args)]
pub struct ConfigArgs {
    /// Output path for generated config.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Show current effective config.
    #[arg(long)]
    pub show: bool,

    /// Validate configuration file.
    #[arg(long)]
    pub validate: bool,

    /// Path to config file to validate or show.
    #[arg(short, long)]
    pub config: Option<PathBuf>,
}

pub fn run_config(args: &ConfigArgs) -> Result<()> {
    if args.validate {
        let config_path = args.config.as_deref();
        match RigConfig::load(config_path) {
            Ok(_) => {
                let path_desc = config_path.map_or_else(
                    || "default locations".to_string(),
                    |p| p.display().to_string(),
                );
                println!("Configuration is valid (loaded from {path_desc})");
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("Configuration validation failed: {e}");
            }
        }
    } else if args.show {
        let config_path = args.config.as_deref();
        let config = RigConfig::load(config_path)?;
        let toml_str = config.to_toml()?;
        println!("{toml_str}");
        Ok(())
    } else {
        let config_str = RigConfig::generate_default_config();

        if let Some(output_path) = &args.output {
            std::fs::write(output_path, &config_str)?;
            println!("Configuration written to {}", output_path.display());
        } else {
            println!("{config_str}");
        }

        Ok(())
    }
}
