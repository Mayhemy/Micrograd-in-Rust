// Tensor micrograd implementation in Rust
pub mod tensor;
pub use tensor::Tensor;
pub mod layers;
pub use layers::{Linear, Conv2d};

#[cfg(test)]
mod tensor_tests;