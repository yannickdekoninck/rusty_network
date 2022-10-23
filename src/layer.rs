use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::network::{Network, NetworkRunState};
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

impl SerializedLayer {
    pub fn add_to_network(self: &Self, network: &mut Network) -> Result<(), &'static str> {
        match self {
            // Convolutional layer
            SerializedLayer::SerialConvolutionalLayer(_) => {
                let mut conv_layer = convolutional::ConvolutionalLayer::empty();
                conv_layer.load_from_serialized(&self)?;
                return network.add_layer(conv_layer);
            }
            // Fully_connected layer
            SerializedLayer::SerialFullyConnectedLayer(_) => {
                let mut fc_layer = fully_connected::FullyConnectedLayer::empty();
                fc_layer.load_from_serialized(&self)?;
                return network.add_layer(fc_layer);
            }
            // Max pooling layer
            SerializedLayer::SerialMaxPoolLayer(_) => {
                let mut mp_layer = max_pool::MaxPoolLayer::empty();
                mp_layer.load_from_serialized(&self)?;
                return network.add_layer(mp_layer);
            }
            // Max pooling layer
            SerializedLayer::SerialSoftMaxLayer(_) => {
                let mut sm_layer = softmax::SoftmaxLayer::empty();
                sm_layer.load_from_serialized(&self)?;
                return network.add_layer(sm_layer);
            }
        }
    }
}

pub trait Layer {
    fn forward(self: &mut Self, input: &Tensor, output: &mut Tensor) -> Result<(), &'static str>;
    // Temporary blank impl so we can test these
    fn backward(
        self: &mut Self,
        input: &Tensor,
        incoming_gradient: &Tensor,
        outgoing_gradient: &mut Tensor,
    ) -> Result<(), &'static str> {
        return Ok(());
    }
    fn apply_gradients(self: &mut Self) {}
    fn clear_gradients(self: &mut Self) {}
    fn switch_to_learning(self: &mut Self) {}
    fn switch_to_inference(self: &mut Self) {}
    fn get_run_mode(self: &Self) -> NetworkRunState {
        return NetworkRunState::Inference;
    }
    fn get_input_shape(self: &Self) -> TensorShape;
    fn get_output_shape(self: &Self) -> TensorShape;
    fn get_name(self: &Self) -> String;
    fn get_serialized(self: &Self) -> SerializedLayer;
    fn load_from_serialized(
        self: &mut Self,
        serial_data: &SerializedLayer,
    ) -> Result<(), &'static str>;
}
