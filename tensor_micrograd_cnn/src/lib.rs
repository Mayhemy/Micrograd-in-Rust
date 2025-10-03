// Tensor micrograd implementation in Rust
pub mod tensor;
pub use tensor::Tensor;

pub mod layers;
pub use layers::{Linear, Conv2d};

pub mod alexnet;
pub use alexnet::{AlexNet};

pub mod train;
pub use train::{S_Gradient_Descent};

pub mod data_loader;

#[cfg(test)]
mod tensor_tests;