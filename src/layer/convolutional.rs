use super::SerializedLayer;
use crate::layer::Layer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ConvolutionalSerialKeys {
    Kernels,
    InputShape,
    Stride,
    Name,
}

pub struct ConvolutionalLayer {
    kernels: Vec<Tensor>,
    stride: u32,
    input_shape: TensorShape,
    output_shape: TensorShape,
    name: String,
}

impl ConvolutionalLayer {
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

        match new_layer.fill_from_state(kernels, input_shape, stride, name) {
            Ok(v) => {
                return Ok(new_layer);
            }
            Err(v) => return Err(v),
        }
    }

    pub fn empty() -> Self {
        let cl = ConvolutionalLayer {
            kernels: vec![],
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
        };
        return cl;
    }

    pub fn fill_from_state(
        self: &mut Self,
        kernels: Vec<Tensor>,
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
        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.stride = stride;
        return Ok(());
    }
}

impl Layer for ConvolutionalLayer {
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor) {
        // Convolution for every output channel
        for i in 0..self.output_shape.dk {
            Tensor::convolution(input, &self.kernels[i as usize], self.stride, output, i);
        }
        return;
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

    fn get_serialized(self: &Self) -> SerializedLayer {
        // Create empty map
        let mut serial_data: HashMap<ConvolutionalSerialKeys, Vec<u8>> = HashMap::new();
        // Serialize elements that need to be serialized
        let serial_kernels = bincode::serialize(&self.kernels).unwrap();
        let serial_input_shape = bincode::serialize(&self.input_shape).unwrap();
        let serial_stride = bincode::serialize(&self.stride).unwrap();
        let serial_name = bincode::serialize(&self.name).unwrap();
        // Add to hashmap
        serial_data.insert(ConvolutionalSerialKeys::Kernels, serial_kernels);
        serial_data.insert(ConvolutionalSerialKeys::InputShape, serial_input_shape);
        serial_data.insert(ConvolutionalSerialKeys::Stride, serial_stride);
        serial_data.insert(ConvolutionalSerialKeys::Name, serial_name);
        // Return wrapped in softmax layer
        return SerializedLayer::SerialConvolutionalLayer(serial_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: SerializedLayer,
    ) -> Result<(), &'static str> {
        return Ok(());
    }
}
