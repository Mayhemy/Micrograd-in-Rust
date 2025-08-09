// Tensor micrograd implementation in Rust
pub mod tensor;
pub use tensor::Tensor;

#[cfg(test)]
mod tensor_tests;