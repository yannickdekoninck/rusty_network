use super::SerializedLayer;
use crate::tensor::helper::TensorShape;
use crate::tensor::{self, Tensor};
use crate::{layer::Layer, network::NetworkRunState};
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ConvolutionalSerialKeys {
    Kernels,
    Biases,
    InputShape,
    Stride,
    Name,
}

pub struct ConvolutionalLayer {
    kernels: Vec<Tensor>,
    biases: Vec<f32>,
    kernel_gradients: Vec<Tensor>,
    bias_gradients: Vec<f32>,
    stride: u32,
    input_shape: TensorShape,
    output_shape: TensorShape,
    name: String,
    run_state: NetworkRunState,
}

impl ConvolutionalLayer {
    fn update_gradient_and_intermediate_tensors(self: &mut Self) {
        match self.run_state {
            NetworkRunState::Inference => {
                // No gradients required
                self.kernel_gradients = vec![];
                self.bias_gradients = vec![];
            }
            NetworkRunState::Training => {
                // Populate the kernel gradients vector
                self.kernel_gradients = vec![];
                for k in &self.kernels {
                    let new_kernel_gradient = Tensor::new(k.get_shape());
                    self.kernel_gradients.push(new_kernel_gradient);
                }
                self.bias_gradients = vec![0.0; self.kernels.len()];
            }
        }
    }
    pub fn new(
        kernel_shape: TensorShape,
        input_shape: TensorShape,
        stride: u32,
        output_depth: u32,
        name: &String,
    ) -> Result<ConvolutionalLayer, &'static str> {
        // Create empty layer
        let mut new_layer = ConvolutionalLayer::empty();

        // Fill up layer with data
        let kernels = vec![Tensor::new(kernel_shape); output_depth as usize];
        let biases = vec![0.0; output_depth as usize];

        match new_layer.fill_from_state(kernels, biases, input_shape, stride, name) {
            Ok(_) => {
                return Ok(new_layer);
            }
            Err(v) => return Err(v),
        }
    }

    pub fn empty() -> Self {
        let cl = ConvolutionalLayer {
            kernels: vec![],
            biases: vec![],
            kernel_gradients: vec![],
            bias_gradients: vec![],
            stride: 1,
            input_shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            output_shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            name: String::from("Empty convolutional layer"),
            run_state: NetworkRunState::Inference,
        };
        return cl;
    }

    pub fn fill_from_state(
        self: &mut Self,
        kernels: Vec<Tensor>,
        biases: Vec<f32>,
        input_shape: TensorShape,
        stride: u32,
        name: &String,
    ) -> Result<(), &'static str> {
        // Check if the kernels all have the same shape
        if kernels.len() < 1 {
            return Err("Kernel list should contain at least 1 kernel");
        }
        let kernel_shape = kernels[0].get_shape();
        for k in &kernels {
            if k.get_shape() != kernel_shape {
                return Err("All kernels in kernel list should have the same shape");
            }
        }
        if !Tensor::does_kernel_stride_fit_image(&input_shape, &kernel_shape, stride) {
            return Err("Kernel does not fit in image with given stride");
        }

        // Calculate output shape
        let output_depth = kernels.len() as u32;

        let (dim0, dim1) = Tensor::get_convolution_dim_fit(&input_shape, &kernel_shape, stride);

        let output_shape = TensorShape {
            di: dim0,
            dj: dim1,
            dk: output_depth,
        };

        // All ready for assignment

        self.name = name.clone();
        self.kernels = kernels;
        self.biases = biases;
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.stride = stride;
        return Ok(());
    }
    fn load_from_serialized_data(
        self: &mut Self,
        data: &HashMap<ConvolutionalSerialKeys, Vec<u8>>,
    ) -> Result<(), &'static str> {
        // Check the correct keys are present
        if !data.contains_key(&ConvolutionalSerialKeys::Name) {
            return Err("Data does not contain --name-- key");
        }
        if !data.contains_key(&ConvolutionalSerialKeys::Kernels) {
            return Err("Data does not contain --kernels-- key");
        }
        if !data.contains_key(&ConvolutionalSerialKeys::Biases) {
            return Err("Data does not contain --biases-- key");
        }
        if !data.contains_key(&ConvolutionalSerialKeys::InputShape) {
            return Err("Data does not contain --input shape-- key");
        }
        if !data.contains_key(&ConvolutionalSerialKeys::Stride) {
            return Err("Data does not contain --stride-- key");
        }

        let kernels_data = data
            .get(&ConvolutionalSerialKeys::Kernels)
            .expect("Cannot access kernels data");
        let kernels: Vec<Tensor> =
            bincode::deserialize(kernels_data).expect("Cannot deserialize kernels data");

        let biases_data = data
            .get(&ConvolutionalSerialKeys::Biases)
            .expect("Cannot access biases data");
        let biases: Vec<f32> =
            bincode::deserialize(biases_data).expect("Cannot deserialize biases data");

        let input_shape_data = data
            .get(&ConvolutionalSerialKeys::InputShape)
            .expect("Cannot access input shape data");
        let input_shape: TensorShape =
            bincode::deserialize(input_shape_data).expect("Cannot deserialize input shape data");

        let stride_data = data
            .get(&ConvolutionalSerialKeys::Stride)
            .expect("Cannot access stride data");
        let stride: u32 =
            bincode::deserialize(&stride_data).expect("Cannot deserialize stride data");

        let name_data = data
            .get(&ConvolutionalSerialKeys::Name)
            .expect("Cannot access name data");
        let name: String = bincode::deserialize(name_data).expect("Cannot deserialize name data");

        return self.fill_from_state(kernels, biases, input_shape, stride, &name);
    }
}

impl Layer for ConvolutionalLayer {
    fn forward(self: &mut Self, input: &Tensor, output: &mut Tensor) -> Result<(), &'static str> {
        // Convolution for every output channel
        for i in 0..self.output_shape.dk {
            tensor::operations::convolution::convolution(
                input,
                &self.kernels[i as usize],
                self.stride,
                output,
                i,
            )?;
        }
        return Ok(());
    }
    fn backward(
        self: &mut Self,
        _input: &Tensor,
        _output: &Tensor,
        _incoming_gradient: &Tensor,
        _outgoing_gradient: &mut Tensor,
    ) -> Result<(), &'static str> {
        return Ok(());
    }
    fn get_output_shape(self: &Self) -> TensorShape {
        return self.output_shape.clone();
    }

    fn get_input_shape(self: &Self) -> TensorShape {
        return self.input_shape.clone();
    }

    fn get_name(self: &Self) -> String {
        return self.name.clone();
    }
    fn get_run_mode(self: &Self) -> NetworkRunState {
        return self.run_state.clone();
    }

    fn clear_gradients(self: &mut Self) {
        // Fill up all
        for v in &mut self.kernel_gradients {
            v.fill_with_value(0.0);
        }
    }

    fn switch_to_inference(self: &mut Self) {
        self.run_state = NetworkRunState::Inference;
        self.update_gradient_and_intermediate_tensors();
    }
    fn switch_to_learning(self: &mut Self) {
        self.run_state = NetworkRunState::Training;
        self.update_gradient_and_intermediate_tensors();
    }

    fn get_serialized(self: &Self) -> SerializedLayer {
        // Create empty map
        let mut serial_data: HashMap<ConvolutionalSerialKeys, Vec<u8>> = HashMap::new();
        // Serialize elements that need to be serialized
        let serial_kernels = bincode::serialize(&self.kernels).unwrap();
        let serial_biases = bincode::serialize(&self.biases).unwrap();
        let serial_input_shape = bincode::serialize(&self.input_shape).unwrap();
        let serial_stride = bincode::serialize(&self.stride).unwrap();
        let serial_name = bincode::serialize(&self.name).unwrap();
        // Add to hashmap
        serial_data.insert(ConvolutionalSerialKeys::Kernels, serial_kernels);
        serial_data.insert(ConvolutionalSerialKeys::Biases, serial_biases);
        serial_data.insert(ConvolutionalSerialKeys::InputShape, serial_input_shape);
        serial_data.insert(ConvolutionalSerialKeys::Stride, serial_stride);
        serial_data.insert(ConvolutionalSerialKeys::Name, serial_name);
        // Return wrapped in softmax layer
        return SerializedLayer::SerialConvolutionalLayer(serial_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: &SerializedLayer,
    ) -> Result<(), &'static str> {
        // Unwrapping the serialized layer and checking it is the correct type
        match serial_data {
            SerializedLayer::SerialConvolutionalLayer(data) => {
                return self.load_from_serialized_data(data);
            }
            _ => {
                return Err("Layer type does not match max pool type");
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let conv_layer = ConvolutionalLayer::new(
            TensorShape {
                di: 2,
                dj: 2,
                dk: 1,
            },
            TensorShape {
                di: 4,
                dj: 4,
                dk: 1,
            },
            2,
            4,
            &String::from("Convolutional layer"),
        )
        .unwrap();

        let serialized_conv_layer = conv_layer.get_serialized();

        let mut new_layer = ConvolutionalLayer::empty();
        assert!(new_layer
            .load_from_serialized(&serialized_conv_layer)
            .is_ok());

        assert_eq!(new_layer.kernels, conv_layer.kernels);
        assert_eq!(new_layer.biases, conv_layer.biases);
        assert_eq!(new_layer.input_shape, conv_layer.input_shape);
        assert_eq!(new_layer.output_shape, conv_layer.output_shape);
        assert_eq!(new_layer.stride, conv_layer.stride);
        assert_eq!(new_layer.name, conv_layer.name);
    }
}
