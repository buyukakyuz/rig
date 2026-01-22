use std::net::SocketAddr;
use std::time::Duration;

pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 60_000;

#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    pub listen_addr: SocketAddr,
    pub heartbeat_interval: Duration,
    pub heartbeat_timeout: Duration,
    pub heartbeat_check_interval: Duration,
    pub max_nodes: usize,
}

impl CoordinatorConfig {
    #[must_use]
    pub fn new(listen_addr: SocketAddr) -> Self {
        Self {
            listen_addr,
            ..Self::default()
        }
    }

    #[must_use]
    pub const fn with_listen_addr(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
        self
    }

    #[must_use]
    pub const fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    #[must_use]
    pub const fn with_heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.heartbeat_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn with_heartbeat_check_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_check_interval = interval;
        self
    }

    #[must_use]
    pub const fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = max;
        self
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        let listen_addr = SocketAddr::from(([0, 0, 0, 0], 50051));

        Self {
            listen_addr,
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
            heartbeat_check_interval: Duration::from_secs(5),
            max_nodes: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.listen_addr.port(), 50051);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(10));
        assert_eq!(config.heartbeat_timeout, Duration::from_secs(30));
        assert_eq!(config.heartbeat_check_interval, Duration::from_secs(5));
        assert_eq!(config.max_nodes, 100);
    }

    #[test]
    fn builder_pattern() {
        let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
        let config = CoordinatorConfig::new(addr)
            .with_heartbeat_interval(Duration::from_secs(5))
            .with_heartbeat_timeout(Duration::from_secs(15))
            .with_max_nodes(50);

        assert_eq!(config.listen_addr, addr);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(5));
        assert_eq!(config.heartbeat_timeout, Duration::from_secs(15));
        assert_eq!(config.max_nodes, 50);
    }
}
