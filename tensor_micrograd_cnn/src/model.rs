use crate::Tensor;
use std::io;
use crate::tensor::Precision;

pub trait Model{
    fn new(num_classes: usize) -> Self;
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn load_parameters(&self, filename: &str) -> bool;
    fn save_parameters(&self, filename: &str, loss: Precision) -> bool;
}