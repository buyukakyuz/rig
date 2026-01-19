use std::time::Duration;

const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_IO_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TcpConfig {
    pub connect_timeout: Duration,
    pub read_timeout: Option<Duration>,
    pub write_timeout: Option<Duration>,
    pub max_message_size: usize,
    pub send_buffer_size: Option<usize>,
    pub recv_buffer_size: Option<usize>,
    pub nodelay: bool,
}

impl TcpConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn with_read_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.read_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn with_write_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.write_timeout = timeout;
        self
    }

    #[must_use]
    pub const fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    #[must_use]
    pub const fn with_send_buffer_size(mut self, size: Option<usize>) -> Self {
        self.send_buffer_size = size;
        self
    }

    #[must_use]
    pub const fn with_recv_buffer_size(mut self, size: Option<usize>) -> Self {
        self.recv_buffer_size = size;
        self
    }

    #[must_use]
    pub const fn with_nodelay(mut self, nodelay: bool) -> Self {
        self.nodelay = nodelay;
        self
    }
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS),
            read_timeout: Some(Duration::from_secs(DEFAULT_IO_TIMEOUT_SECS)),
            write_timeout: Some(Duration::from_secs(DEFAULT_IO_TIMEOUT_SECS)),
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            send_buffer_size: None,
            recv_buffer_size: None,
            nodelay: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let config = TcpConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.read_timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.write_timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.max_message_size, 64 * 1024 * 1024);
        assert!(config.nodelay);
    }

    #[test]
    fn builder_methods_work() {
        let config = TcpConfig::new()
            .with_connect_timeout(Duration::from_secs(5))
            .with_read_timeout(None)
            .with_max_message_size(1024 * 1024)
            .with_nodelay(false);

        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert!(config.read_timeout.is_none());
        assert_eq!(config.max_message_size, 1024 * 1024);
        assert!(!config.nodelay);
    }
}
