use std::ops::Range;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    EvenSplit { num_stages: usize },
    Manual { ranges: Vec<Range<usize>> },
    MemoryBalanced { node_memory_bytes: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PartitioningError {
    #[error("at least one stage is required")]
    ZeroStages,

    #[error("too many stages ({num_stages}) for {num_layers} layers")]
    TooManyStages {
        num_stages: usize,
        num_layers: usize,
    },

    #[error("manual partitioning requires at least one range")]
    EmptyRanges,

    #[error("ranges must start at layer 0, but first range starts at {start}")]
    RangesNotStartingAtZero { start: usize },

    #[error("ranges are not contiguous: gap between layer {prev_end} and {next_start}")]
    NonContiguousRanges { prev_end: usize, next_start: usize },

    #[error("ranges end at layer {ranges_end} but model has {num_layers} layers")]
    RangesNotCoveringAll {
        ranges_end: usize,
        num_layers: usize,
    },

    #[error("range {index} is empty: {start}..{end}")]
    EmptyRange {
        index: usize,
        start: usize,
        end: usize,
    },

    #[error("memory balanced partitioning requires at least one node")]
    NoNodeMemory,

    #[error("total memory across all nodes is zero")]
    ZeroTotalMemory,
}

impl PartitioningStrategy {
    pub fn compute_ranges(
        &self,
        num_layers: usize,
        _num_nodes: usize,
    ) -> Result<Vec<Range<usize>>, PartitioningError> {
        match self {
            Self::EvenSplit { num_stages } => Self::compute_even_split(num_layers, *num_stages),
            Self::Manual { ranges } => {
                Self::validate_manual_ranges(ranges, num_layers)?;
                Ok(ranges.clone())
            }
            Self::MemoryBalanced { node_memory_bytes } => {
                Self::compute_memory_balanced(num_layers, node_memory_bytes)
            }
        }
    }

    fn compute_even_split(
        num_layers: usize,
        num_stages: usize,
    ) -> Result<Vec<Range<usize>>, PartitioningError> {
        if num_stages == 0 {
            return Err(PartitioningError::ZeroStages);
        }

        if num_stages > num_layers {
            return Err(PartitioningError::TooManyStages {
                num_stages,
                num_layers,
            });
        }

        let base = num_layers / num_stages;
        let remainder = num_layers % num_stages;
        let mut ranges = Vec::with_capacity(num_stages);
        let mut start = 0;

        for i in 0..num_stages {
            let extra = usize::from(i < remainder);
            let end = start + base + extra;
            ranges.push(start..end);
            start = end;
        }

        Ok(ranges)
    }

    fn validate_manual_ranges(
        ranges: &[Range<usize>],
        num_layers: usize,
    ) -> Result<(), PartitioningError> {
        if ranges.is_empty() {
            return Err(PartitioningError::EmptyRanges);
        }

        if ranges[0].start != 0 {
            return Err(PartitioningError::RangesNotStartingAtZero {
                start: ranges[0].start,
            });
        }

        for (i, range) in ranges.iter().enumerate() {
            if range.start >= range.end {
                return Err(PartitioningError::EmptyRange {
                    index: i,
                    start: range.start,
                    end: range.end,
                });
            }

            if i > 0 {
                let prev_end = ranges[i - 1].end;
                if range.start != prev_end {
                    return Err(PartitioningError::NonContiguousRanges {
                        prev_end,
                        next_start: range.start,
                    });
                }
            }
        }

        let last_end = ranges.last().map_or(0, |r| r.end);
        if last_end != num_layers {
            return Err(PartitioningError::RangesNotCoveringAll {
                ranges_end: last_end,
                num_layers,
            });
        }

        Ok(())
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn compute_memory_balanced(
        num_layers: usize,
        node_memory_bytes: &[u64],
    ) -> Result<Vec<Range<usize>>, PartitioningError> {
        if node_memory_bytes.is_empty() {
            return Err(PartitioningError::NoNodeMemory);
        }

        let total_memory: u64 = node_memory_bytes.iter().sum();
        if total_memory == 0 {
            return Err(PartitioningError::ZeroTotalMemory);
        }

        let num_nodes = node_memory_bytes.len();
        if num_nodes > num_layers {
            return Err(PartitioningError::TooManyStages {
                num_stages: num_nodes,
                num_layers,
            });
        }

        let mut ranges = Vec::with_capacity(num_nodes);
        let mut start = 0;
        let mut allocated_layers = 0;

        for (i, &memory) in node_memory_bytes.iter().enumerate() {
            let is_last = i == num_nodes - 1;

            if is_last {
                ranges.push(start..num_layers);
            } else {
                let proportion = memory as f64 / total_memory as f64;
                let target_layers = (num_layers as f64 * proportion).round() as usize;

                let remaining = num_layers - allocated_layers;
                let remaining_nodes = num_nodes - i;
                let min_layers = 1;
                let max_layers = remaining.saturating_sub(remaining_nodes - 1);
                let layers = target_layers.clamp(min_layers, max_layers);

                let end = start + layers;
                ranges.push(start..end);
                start = end;
                allocated_layers += layers;
            }
        }

        Ok(ranges)
    }

    #[must_use]
    pub const fn even_split(num_stages: usize) -> Self {
        Self::EvenSplit { num_stages }
    }

    #[must_use]
    pub const fn manual(ranges: Vec<Range<usize>>) -> Self {
        Self::Manual { ranges }
    }

    #[must_use]
    pub const fn memory_balanced(node_memory_bytes: Vec<u64>) -> Self {
        Self::MemoryBalanced { node_memory_bytes }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn even_split_basic() {
        let strategy = PartitioningStrategy::even_split(3);
        let ranges = strategy.compute_ranges(12, 3).unwrap();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..4);
        assert_eq!(ranges[1], 4..8);
        assert_eq!(ranges[2], 8..12);
    }

    #[test]
    fn even_split_with_remainder() {
        let strategy = PartitioningStrategy::even_split(3);
        let ranges = strategy.compute_ranges(10, 3).unwrap();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..4);
        assert_eq!(ranges[1], 4..7);
        assert_eq!(ranges[2], 7..10);
    }

    #[test]
    fn even_split_single_stage() {
        let strategy = PartitioningStrategy::even_split(1);
        let ranges = strategy.compute_ranges(32, 1).unwrap();

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..32);
    }

    #[test]
    fn even_split_zero_stages_error() {
        let strategy = PartitioningStrategy::even_split(0);
        let result = strategy.compute_ranges(12, 3);

        assert!(matches!(result, Err(PartitioningError::ZeroStages)));
    }

    #[test]
    fn even_split_too_many_stages_error() {
        let strategy = PartitioningStrategy::even_split(10);
        let result = strategy.compute_ranges(5, 10);

        assert!(matches!(
            result,
            Err(PartitioningError::TooManyStages {
                num_stages: 10,
                num_layers: 5
            })
        ));
    }

    #[test]
    fn manual_valid_ranges() {
        let strategy = PartitioningStrategy::manual(vec![0..4, 4..8, 8..12]);
        let ranges = strategy.compute_ranges(12, 3).unwrap();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..4);
        assert_eq!(ranges[1], 4..8);
        assert_eq!(ranges[2], 8..12);
    }

    #[test]
    fn manual_empty_ranges_error() {
        let strategy = PartitioningStrategy::manual(vec![]);
        let result = strategy.compute_ranges(12, 3);

        assert!(matches!(result, Err(PartitioningError::EmptyRanges)));
    }

    #[test]
    fn manual_not_starting_at_zero_error() {
        let strategy = PartitioningStrategy::manual(vec![1..5, 5..10]);
        let result = strategy.compute_ranges(10, 2);

        assert!(matches!(
            result,
            Err(PartitioningError::RangesNotStartingAtZero { start: 1 })
        ));
    }

    #[test]
    fn manual_non_contiguous_error() {
        let strategy = PartitioningStrategy::manual(vec![0..4, 5..10]);
        let result = strategy.compute_ranges(10, 2);

        assert!(matches!(
            result,
            Err(PartitioningError::NonContiguousRanges {
                prev_end: 4,
                next_start: 5
            })
        ));
    }

    #[test]
    fn manual_not_covering_all_error() {
        let strategy = PartitioningStrategy::manual(vec![0..4, 4..8]);
        let result = strategy.compute_ranges(12, 2);

        assert!(matches!(
            result,
            Err(PartitioningError::RangesNotCoveringAll {
                ranges_end: 8,
                num_layers: 12
            })
        ));
    }

    #[test]
    fn manual_empty_range_error() {
        let strategy = PartitioningStrategy::manual(vec![0..4, 4..4, 4..10]);
        let result = strategy.compute_ranges(10, 3);

        assert!(matches!(
            result,
            Err(PartitioningError::EmptyRange {
                index: 1,
                start: 4,
                end: 4
            })
        ));
    }

    #[test]
    fn memory_balanced_equal_memory() {
        let strategy = PartitioningStrategy::memory_balanced(vec![1000, 1000, 1000]);
        let ranges = strategy.compute_ranges(12, 3).unwrap();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..4);
        assert_eq!(ranges[1], 4..8);
        assert_eq!(ranges[2], 8..12);
    }

    #[test]
    fn memory_balanced_unequal_memory() {
        let strategy = PartitioningStrategy::memory_balanced(vec![2000, 1000, 1000]);
        let ranges = strategy.compute_ranges(12, 3).unwrap();

        assert_eq!(ranges.len(), 3);
        let total: usize = ranges.iter().map(|r| r.end - r.start).sum();
        assert_eq!(total, 12);
    }

    #[test]
    fn memory_balanced_no_nodes_error() {
        let strategy = PartitioningStrategy::memory_balanced(vec![]);
        let result = strategy.compute_ranges(12, 0);

        assert!(matches!(result, Err(PartitioningError::NoNodeMemory)));
    }

    #[test]
    fn memory_balanced_zero_memory_error() {
        let strategy = PartitioningStrategy::memory_balanced(vec![0, 0, 0]);
        let result = strategy.compute_ranges(12, 3);

        assert!(matches!(result, Err(PartitioningError::ZeroTotalMemory)));
    }

    #[test]
    fn memory_balanced_too_many_nodes_error() {
        let strategy = PartitioningStrategy::memory_balanced(vec![1000, 1000, 1000, 1000, 1000]);
        let result = strategy.compute_ranges(3, 5);

        assert!(matches!(
            result,
            Err(PartitioningError::TooManyStages {
                num_stages: 5,
                num_layers: 3
            })
        ));
    }

    #[test]
    fn serialization_roundtrip() {
        let strategies = vec![
            PartitioningStrategy::even_split(4),
            PartitioningStrategy::manual(vec![0..10, 10..20]),
            PartitioningStrategy::memory_balanced(vec![1024, 2048]),
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let recovered: PartitioningStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, recovered);
        }
    }
}
