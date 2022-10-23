use super::SerializedLayer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;
use crate::{layer::Layer, network::NetworkRunState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum MaxPoolSerialKeys {
    InputShape,
    MaskShape,
    Stride,
    Name,
}

pub struct MaxPoolLayer {
    input_shape: TensorShape,
    mask_shape: TensorShape,
    output_shape: TensorShape,
    stride: u32,
    name: String,
    run_state: NetworkRunState,
    max_pool_origin: Vec<u32>,
}

impl MaxPoolLayer {
    pub fn new(
        mask_shape: TensorShape,
        input_shape: TensorShape,
        stride: u32,
        name: &String,
    ) -> Result<MaxPoolLayer, &'static str> {
        let mut cl = MaxPoolLayer::empty();
        match cl.fill_from_state(input_shape, mask_shape, stride, name) {
            Ok(_) => return Ok(cl),
            Err(msg) => return Err(msg),
        };
    }
    pub fn empty() -> Self {
        let mp = MaxPoolLayer {
            input_shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            mask_shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            output_shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            stride: 1,
            name: String::from("Empty max pool layer"),
            run_state: NetworkRunState::Inference,
            max_pool_origin: vec![],
        };
        return mp;
    }

    fn update_gradient_and_intermediate_tensors(self: &mut Self) {
        match self.run_state {
            NetworkRunState::Inference => {
                // Clean up all training related tensors
                self.max_pool_origin = vec![];
            }
            NetworkRunState::Training => {
                // Allocate a vector of indices to store the origin of the max pool
                self.max_pool_origin = vec![0; self.output_shape.total_size() as usize];
                // Fill up all tensors with 0.0
                self.clear_gradients();
            }
        }
    }

    pub fn fill_from_state(
        self: &mut Self,
        input_shape: TensorShape,
        mask_shape: TensorShape,
        stride: u32,
        name: &String,
    ) -> Result<(), &'static str> {
        if !Tensor::does_kernel_stride_fit_image(&input_shape, &mask_shape, stride) {
            return Err("Mask does not fit in image with given stride");
        }
        let (dim0, dim1) = Tensor::get_convolution_dim_fit(&input_shape, &mask_shape, stride);

        let output_shape = TensorShape {
            di: dim0,
            dj: dim1,
            dk: input_shape.dk,
        };

        self.input_shape = input_shape;
        self.output_shape = output_shape;
        self.mask_shape = mask_shape;
        self.stride = stride;
        self.name = name.clone();

        return Ok(());
    }

    fn load_from_serialized_data(
        self: &mut Self,
        data: &HashMap<MaxPoolSerialKeys, Vec<u8>>,
    ) -> Result<(), &'static str> {
        // Check the correct keys are present
        if !data.contains_key(&MaxPoolSerialKeys::Name) {
            return Err("Data does not contain --name-- key");
        }
        if !data.contains_key(&MaxPoolSerialKeys::MaskShape) {
            return Err("Data does not contain --mask shape-- key");
        }
        if !data.contains_key(&MaxPoolSerialKeys::InputShape) {
            return Err("Data does not contain --input shape-- key");
        }
        if !data.contains_key(&MaxPoolSerialKeys::Stride) {
            return Err("Data does not contain --stride-- key");
        }

        let mask_shape_data = data
            .get(&MaxPoolSerialKeys::MaskShape)
            .expect("Cannot access mask shape data");
        let mask_shape: TensorShape =
            bincode::deserialize(mask_shape_data).expect("Cannot deserialize mask shape data");
        let input_shape_data = data
            .get(&MaxPoolSerialKeys::InputShape)
            .expect("Cannot access input shape data");
        let input_shape: TensorShape =
            bincode::deserialize(input_shape_data).expect("Cannot deserialize input shape data");
        let stride_data = data
            .get(&MaxPoolSerialKeys::Stride)
            .expect("Cannot access stride data");
        let stride: u32 =
            bincode::deserialize(&stride_data).expect("Cannot deserialize stride data");

        let name_data = data
            .get(&MaxPoolSerialKeys::Name)
            .expect("Cannot access name data");
        let name: String = bincode::deserialize(name_data).expect("Cannot deserialize name data");
        return self.fill_from_state(input_shape, mask_shape, stride, &name);
    }
}

impl Layer for MaxPoolLayer {
    fn forward(self: &mut Self, input: &Tensor, output: &mut Tensor) -> Result<(), &'static str> {
        match self.run_state {
            NetworkRunState::Inference => {
                Tensor::max_pool(input, &self.mask_shape, self.stride, output);
            }
            NetworkRunState::Training => {
                Tensor::max_pool_track_origin(
                    input,
                    &self.mask_shape,
                    self.stride,
                    output,
                    &mut self.max_pool_origin,
                )?;
            }
        }
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

    fn get_serialized(self: &Self) -> SerializedLayer {
        // Create empty map
        let mut serial_data: HashMap<MaxPoolSerialKeys, Vec<u8>> = HashMap::new();
        // Serialize elements that need to be serialized
        let serial_mask_shape = bincode::serialize(&self.mask_shape).unwrap();
        let serial_input_shape = bincode::serialize(&self.input_shape).unwrap();
        let serial_stride = bincode::serialize(&self.stride).unwrap();
        let serial_name = bincode::serialize(&self.name).unwrap();
        // Add to hashmap
        serial_data.insert(MaxPoolSerialKeys::MaskShape, serial_mask_shape);
        serial_data.insert(MaxPoolSerialKeys::InputShape, serial_input_shape);
        serial_data.insert(MaxPoolSerialKeys::Stride, serial_stride);
        serial_data.insert(MaxPoolSerialKeys::Name, serial_name);
        // Return wrapped in softmax layer
        return SerializedLayer::SerialMaxPoolLayer(serial_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: &SerializedLayer,
    ) -> Result<(), &'static str> {
        // Unwrapping the serialized layer and checking it is the correct type
        match serial_data {
            SerializedLayer::SerialMaxPoolLayer(data) => {
                return self.load_from_serialized_data(data);
            }
            _ => {
                return Err("Layer type does not match max pool type");
            }
        }
    }

    fn clear_gradients(self: &mut Self) {
        for v in self.max_pool_origin.iter_mut() {
            *v = 0;
        }
    }

    fn switch_to_inference(self: &mut Self) {
        // Update run state
        self.run_state = NetworkRunState::Inference;
        // Empty gradients
        self.update_gradient_and_intermediate_tensors();
    }

    fn switch_to_learning(self: &mut Self) {
        // Update run state
        self.run_state = NetworkRunState::Training;
        // Create tensors with the correct shapes
        self.update_gradient_and_intermediate_tensors();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let mpl = MaxPoolLayer::new(
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
            1,
            &String::from("Max pooling"),
        )
        .unwrap();
        let serialized_mpl = mpl.get_serialized();

        let mut mpl2 = MaxPoolLayer::new(
            TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            1,
            &String::from("Empty layer"),
        )
        .unwrap();

        assert!(mpl2.load_from_serialized(&serialized_mpl).is_ok());

        assert_eq!(mpl2.name, String::from("Max pooling"));
        assert_eq!(
            mpl2.mask_shape,
            TensorShape {
                di: 2,
                dj: 2,
                dk: 1,
            },
        );
        assert_eq!(
            mpl2.input_shape,
            TensorShape {
                di: 4,
                dj: 4,
                dk: 1,
            },
        );
        assert_eq!(mpl2.stride, 1);
    }
}
