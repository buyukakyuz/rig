use crate::traits::{Partition, Tokenizer};

pub struct LoadedPartition {
    partition: Box<dyn Partition>,
    tokenizer: Option<Box<dyn Tokenizer>>,
}

impl LoadedPartition {
    pub fn new(partition: Box<dyn Partition>, tokenizer: Option<Box<dyn Tokenizer>>) -> Self {
        Self {
            partition,
            tokenizer,
        }
    }

    pub fn without_tokenizer(partition: Box<dyn Partition>) -> Self {
        Self {
            partition,
            tokenizer: None,
        }
    }

    pub fn partition(&self) -> &dyn Partition {
        &*self.partition
    }

    pub fn partition_mut(&mut self) -> &mut dyn Partition {
        &mut *self.partition
    }

    pub fn into_partition(self) -> Box<dyn Partition> {
        self.partition
    }

    pub fn into_parts(self) -> (Box<dyn Partition>, Option<Box<dyn Tokenizer>>) {
        (self.partition, self.tokenizer)
    }

    pub fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tokenizer.as_deref()
    }

    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }
}
