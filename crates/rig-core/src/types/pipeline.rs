use std::ops::Range;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::types::id::{ModelId, NodeId, PipelineId, StageId};
use crate::types::node::Address;
use crate::types::tensor::DType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: ModelId,
    pub path: PathBuf,
    pub num_layers: usize,
    pub hidden_dim: usize,
}

impl ModelSpec {
    #[must_use]
    pub fn new(
        model_id: ModelId,
        path: impl Into<PathBuf>,
        num_layers: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            model_id,
            path: path.into(),
            num_layers,
            hidden_dim,
        }
    }

    #[must_use]
    pub const fn with_path(
        model_id: ModelId,
        path: PathBuf,
        num_layers: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            model_id,
            path,
            num_layers,
            hidden_dim,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.model_id.name
    }

    #[must_use]
    pub fn version(&self) -> &str {
        &self.model_id.version
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionSpec {
    pub layer_range: Range<usize>,
    pub dtype: DType,
}

impl PartitionSpec {
    #[must_use]
    pub const fn new(layer_range: Range<usize>, dtype: DType) -> Self {
        Self { layer_range, dtype }
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layer_range.len()
    }

    #[must_use]
    pub fn contains_layer(&self, layer: usize) -> bool {
        self.layer_range.contains(&layer)
    }

    #[must_use]
    pub const fn is_first(&self) -> bool {
        self.layer_range.start == 0
    }

    #[must_use]
    pub const fn is_last(&self, total_layers: usize) -> bool {
        self.layer_range.end == total_layers
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub num_stages: usize,
    pub dtype: DType,
}

impl PipelineConfig {
    #[must_use]
    pub fn new(
        model_id: ModelId,
        model_path: impl Into<PathBuf>,
        num_stages: usize,
        dtype: DType,
    ) -> Self {
        Self {
            model_id,
            model_path: model_path.into(),
            num_stages,
            dtype,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerAddress {
    pub node_id: NodeId,
    pub addresses: Vec<Address>,
}

impl PeerAddress {
    #[must_use]
    pub const fn new(node_id: NodeId, addresses: Vec<Address>) -> Self {
        Self { node_id, addresses }
    }

    #[must_use]
    pub fn with_address(node_id: NodeId, address: Address) -> Self {
        Self {
            node_id,
            addresses: vec![address],
        }
    }

    #[must_use]
    pub fn first_address(&self) -> Option<&Address> {
        self.addresses.first()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Neighbors {
    pub prev: Option<PeerAddress>,
    pub next: Option<PeerAddress>,
}

impl Neighbors {
    #[must_use]
    pub const fn none() -> Self {
        Self {
            prev: None,
            next: None,
        }
    }

    #[must_use]
    pub const fn first_stage(next: PeerAddress) -> Self {
        Self {
            prev: None,
            next: Some(next),
        }
    }

    #[must_use]
    pub const fn last_stage(prev: PeerAddress) -> Self {
        Self {
            prev: Some(prev),
            next: None,
        }
    }

    #[must_use]
    pub const fn middle_stage(prev: PeerAddress, next: PeerAddress) -> Self {
        Self {
            prev: Some(prev),
            next: Some(next),
        }
    }

    #[must_use]
    pub const fn is_first_stage(&self) -> bool {
        self.prev.is_none()
    }

    #[must_use]
    pub const fn is_last_stage(&self) -> bool {
        self.next.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub pipeline_id: PipelineId,
    pub stage_id: StageId,
    pub layer_range: Range<usize>,
    pub neighbors: Neighbors,
}

impl Assignment {
    #[must_use]
    pub const fn new(
        pipeline_id: PipelineId,
        stage_id: StageId,
        layer_range: Range<usize>,
        neighbors: Neighbors,
    ) -> Self {
        Self {
            pipeline_id,
            stage_id,
            layer_range,
            neighbors,
        }
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layer_range.len()
    }

    #[must_use]
    pub const fn is_first_stage(&self) -> bool {
        self.neighbors.is_first_stage()
    }

    #[must_use]
    pub const fn is_last_stage(&self) -> bool {
        self.neighbors.is_last_stage()
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn partition_spec_num_layers() {
        let spec = PartitionSpec::new(10..30, DType::F16);
        assert_eq!(spec.num_layers(), 20);
        assert!(spec.contains_layer(15));
        assert!(!spec.contains_layer(5));
        assert!(!spec.contains_layer(30));
    }

    #[test]
    fn partition_spec_first_last() {
        let first = PartitionSpec::new(0..20, DType::F16);
        assert!(first.is_first());
        assert!(!first.is_last(80));

        let last = PartitionSpec::new(60..80, DType::F16);
        assert!(!last.is_first());
        assert!(last.is_last(80));
    }

    #[test]
    fn neighbors_stages() {
        let first = Neighbors::first_stage(PeerAddress::new(NodeId::new(), vec![]));
        assert!(first.is_first_stage());
        assert!(!first.is_last_stage());

        let last = Neighbors::last_stage(PeerAddress::new(NodeId::new(), vec![]));
        assert!(!last.is_first_stage());
        assert!(last.is_last_stage());

        let middle = Neighbors::middle_stage(
            PeerAddress::new(NodeId::new(), vec![]),
            PeerAddress::new(NodeId::new(), vec![]),
        );
        assert!(!middle.is_first_stage());
        assert!(!middle.is_last_stage());
    }

    #[test]
    fn assignment_num_layers() {
        let assignment = Assignment::new(
            PipelineId::new(),
            StageId::new(1),
            20..40,
            Neighbors::none(),
        );
        assert_eq!(assignment.num_layers(), 20);
    }

    #[test]
    fn model_spec_accessors() {
        let spec = ModelSpec::new(
            ModelId::new("llama-7b", "q4_K_M"),
            "/models/llama.gguf",
            32,
            4096,
        );
        assert_eq!(spec.name(), "llama-7b");
        assert_eq!(spec.version(), "q4_K_M");
        assert_eq!(spec.num_layers, 32);
    }

    #[test]
    fn pipeline_config_serialization_roundtrip() {
        let config = PipelineConfig::new(
            ModelId::new("test-model", "v1"),
            "/path/to/model",
            4,
            DType::F16,
        );
        let json =
            serde_json::to_string(&config).unwrap_or_else(|e| panic!("serialize failed: {e}"));
        let recovered: PipelineConfig =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize failed: {e}"));
        assert_eq!(config.model_id, recovered.model_id);
        assert_eq!(config.num_stages, recovered.num_stages);
    }
}
