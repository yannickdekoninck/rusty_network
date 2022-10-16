use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tensor::{helper::TensorShape, Tensor};

pub mod convolutional;
pub mod fully_connected;
pub mod max_pool;
pub mod softmax;

#[derive(Serialize, Deserialize)]
pub enum SerializedLayer {
    SerialConvolutionalLayer(HashMap<convolutional::ConvolutionalSerialKeys, Vec<u8>>),
    SerialFullyConnectedLayer(HashMap<fully_connected::FullyConnectedSerialKeys, Vec<u8>>),
    SerialMaxPoolLayer(HashMap<max_pool::MaxPoolSerialKeys, Vec<u8>>),
    SerialSoftMaxLayer(HashMap<softmax::SoftMaxSerialKeys, Vec<u8>>),
}

pub trait Layer {
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor);
    fn get_input_shape(self: &Self) -> TensorShape;
    fn get_output_shape(self: &Self) -> TensorShape;
    fn get_name(self: &Self) -> String;
    fn get_serialized(self: &Self) -> SerializedLayer;
    fn load_from_serialized(
        self: &mut Self,
        serial_data: SerializedLayer,
    ) -> Result<(), &'static str>;
}
