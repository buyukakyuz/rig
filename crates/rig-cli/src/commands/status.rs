use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Args;
use rig_core::NodeStatus;

use crate::client::CliClient;
use crate::output::{NodeOutput, PipelineOutput, StageOutput, StatusOutput};

#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Coordinator address.
    #[arg(long, env = "RIG_COORDINATOR_ADDR", default_value = "127.0.0.1:50051")]
    pub coordinator: String,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[allow(clippy::too_many_lines)]
pub async fn run_status(args: StatusArgs) -> Result<()> {
    let coordinator_addr: SocketAddr = args
        .coordinator
        .parse()
        .context("Invalid coordinator address")?;

    let client = CliClient::connect(coordinator_addr).await?;
    let status = client.get_status().await?;

    if args.json {
        let output = StatusOutput {
            total_nodes: status.total_nodes,
            healthy_nodes: status.healthy_nodes,
            total_pipelines: status.total_pipelines,
            ready_pipelines: status.ready_pipelines,
            nodes: status
                .nodes
                .iter()
                .map(|n| NodeOutput {
                    node_id: n.node_id.to_string(),
                    addresses: n.addresses.clone(),
                    status: format!("{:?}", n.status),
                    runtime_type: n.runtime_type.clone(),
                    vram_bytes: n.vram_bytes,
                })
                .collect(),
            pipelines: status
                .pipelines
                .iter()
                .map(|p| PipelineOutput {
                    pipeline_id: p.pipeline_id.to_string(),
                    model_id: format!("{}:{}", p.model_id.name, p.model_id.version),
                    status: p.status.clone(),
                    stages: p
                        .stages
                        .iter()
                        .map(|s| StageOutput {
                            stage_id: s.stage_id,
                            node_id: s.node_id.to_string(),
                            layer_range: format!("{}-{}", s.layer_start, s.layer_end),
                            ready: s.ready,
                        })
                        .collect(),
                })
                .collect(),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Cluster Status");
        println!("==============");
        println!();
        println!(
            "Nodes:     {}/{} healthy",
            status.healthy_nodes, status.total_nodes
        );
        println!(
            "Pipelines: {}/{} ready",
            status.ready_pipelines, status.total_pipelines
        );

        if args.verbose || !status.nodes.is_empty() {
            println!();
            println!("Nodes:");
            println!("------");
            if status.nodes.is_empty() {
                println!("  (none)");
            } else {
                for node in &status.nodes {
                    let status_str = match &node.status {
                        NodeStatus::Healthy => "healthy".to_string(),
                        NodeStatus::Unhealthy { reason } => format!("unhealthy: {reason}"),
                        NodeStatus::Draining => "draining".to_string(),
                        NodeStatus::Offline => "offline".to_string(),
                    };
                    let addr = node.addresses.first().map_or("-", String::as_str);
                    println!("  {} ({}) - {}", node.node_id, addr, status_str);
                    if args.verbose {
                        if let Some(runtime) = &node.runtime_type {
                            println!("    Runtime: {runtime}");
                        }
                        if let Some(vram) = node.vram_bytes {
                            #[allow(clippy::cast_precision_loss)]
                            let vram_gb = vram as f64 / (1024.0 * 1024.0 * 1024.0);
                            println!("    VRAM: {vram_gb:.1} GB");
                        }
                    }
                }
            }
        }

        if args.verbose || !status.pipelines.is_empty() {
            println!();
            println!("Pipelines:");
            println!("----------");
            if status.pipelines.is_empty() {
                println!("  (none)");
            } else {
                for pipeline in &status.pipelines {
                    println!(
                        "  {} - {}:{} [{}]",
                        pipeline.pipeline_id,
                        pipeline.model_id.name,
                        pipeline.model_id.version,
                        pipeline.status
                    );
                    if args.verbose {
                        for stage in &pipeline.stages {
                            let ready = if stage.ready { "ready" } else { "pending" };
                            println!(
                                "    Stage {}: layers {}-{} on {} ({})",
                                stage.stage_id,
                                stage.layer_start,
                                stage.layer_end,
                                stage.node_id,
                                ready
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
