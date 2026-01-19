use tokio::sync::broadcast;

use crate::error::{ClusterError, CoordError, DiscoveryError, HealthError};
use crate::types::{
    Assignment, ClusterEvent, ClusterState, JoinResult, Neighbors, NodeId, NodeInfo, NodeStatus,
    PeerAddress, WorkResult,
};

pub trait ClusterMembership: Send + Sync {
    fn join(
        &self,
        self_info: NodeInfo,
    ) -> impl std::future::Future<Output = Result<JoinResult, ClusterError>> + Send;
    fn leave(&self) -> impl std::future::Future<Output = Result<(), ClusterError>> + Send;
    fn cluster_state(
        &self,
    ) -> impl std::future::Future<Output = Result<ClusterState, ClusterError>> + Send;
    fn subscribe(&self) -> broadcast::Receiver<ClusterEvent>;
}

pub trait WorkCoordination: Send + Sync {
    fn get_assignment(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<Assignment>, CoordError>> + Send;
    fn report_ready(&self) -> impl std::future::Future<Output = Result<(), CoordError>> + Send;
    fn report_complete(
        &self,
        result: WorkResult,
    ) -> impl std::future::Future<Output = Result<(), CoordError>> + Send;
}

pub trait PeerDiscovery: Send + Sync {
    fn resolve(
        &self,
        node_id: &NodeId,
    ) -> impl std::future::Future<Output = Result<PeerAddress, DiscoveryError>> + Send;
    fn get_neighbors(
        &self,
    ) -> impl std::future::Future<Output = Result<Neighbors, DiscoveryError>> + Send;
}

pub trait HealthReporter: Send + Sync {
    fn start(&self) -> impl std::future::Future<Output = Result<(), HealthError>> + Send;
    fn stop(&self) -> impl std::future::Future<Output = Result<(), HealthError>> + Send;
    fn report(
        &self,
        status: NodeStatus,
    ) -> impl std::future::Future<Output = Result<(), HealthError>> + Send;
}
