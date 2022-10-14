use crate::tensor::{helper::TensorShape, Tensor};

pub mod convolutional;
pub mod fully_connected;
pub mod max_pool;

pub trait Layer {
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor);
    fn get_input_shape(self: &Self) -> TensorShape;
    fn get_output_shape(self: &Self) -> TensorShape;
    fn get_name(self: &Self) -> String;
}
