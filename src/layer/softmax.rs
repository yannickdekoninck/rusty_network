use serde::{Deserialize, Serialize};
use std::cmp::Eq;
use std::collections::HashMap;

use crate::layer::Layer;
use crate::network::NetworkRunState;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;

use super::SerializedLayer;

// using an enum for defining keys in serialized data
#[derive(Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum SoftMaxSerialKeys {
    Shape,
    Name,
}

pub struct SoftmaxLayer {
    shape: TensorShape,
    name: String,
    intermediate1: Tensor,
    intermediate2: Tensor,
    run_state: NetworkRunState,
}

impl SoftmaxLayer {
    pub fn new(shape: TensorShape, name: &String) -> Result<SoftmaxLayer, &'static str> {
        let mut sm = SoftmaxLayer::empty();
        sm.fill_from_state(shape, name)?;
        return Ok(sm);
    }

    pub fn empty() -> Self {
        let sm = SoftmaxLayer {
            shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            name: String::from("Empty softmax layer"),
            intermediate1: Tensor::empty(),
            intermediate2: Tensor::empty(),
            run_state: NetworkRunState::Inference,
        };

        return sm;
    }

    fn update_gradient_and_intermediate_tensors(self: &mut Self) {
        match self.run_state {
            NetworkRunState::Inference => {
                // Clean up all training related tensors

                self.intermediate1 = Tensor::empty();
                self.intermediate2 = Tensor::empty();
            }
            NetworkRunState::Training => {
                // Allocate all tensors required for training
                self.intermediate1 = Tensor::new(TensorShape::new_1d(1));
                self.intermediate2 = Tensor::new(self.shape);
                // Fill up all tensors with 0.0
                self.clear_gradients();
            }
        }
    }

    pub fn fill_from_state(
        self: &mut Self,
        shape: TensorShape,
        name: &String,
    ) -> Result<(), &'static str> {
        self.shape = shape;
        self.name = name.clone();
        self.update_gradient_and_intermediate_tensors();
        return Ok(());
    }

    fn load_from_serialized_data(
        self: &mut Self,
        data: &HashMap<SoftMaxSerialKeys, Vec<u8>>,
    ) -> Result<(), &'static str> {
        // Check the correct keys are present
        if !data.contains_key(&SoftMaxSerialKeys::Name) {
            return Err("Data does not contain --name-- key");
        }
        if !data.contains_key(&SoftMaxSerialKeys::Shape) {
            return Err("Data does not contain --shape-- key");
        }

        let shape_data = data
            .get(&SoftMaxSerialKeys::Shape)
            .expect("Cannot access shape data");
        self.shape = bincode::deserialize(shape_data).expect("Cannot deserialize shape data");

        let name_data = data
            .get(&SoftMaxSerialKeys::Name)
            .expect("Cannot access name data");
        self.name = bincode::deserialize(name_data).expect("Cannot deserialize name data");

        return Ok(());
    }
}

impl Layer for SoftmaxLayer {
    fn forward(self: &mut Self, input: &Tensor, output: &mut Tensor) -> Result<(), &'static str> {
        Tensor::softmax(input, output);
        return Ok(());
    }
    fn backward(
        self: &mut Self,
        _input: &Tensor,
        output: &Tensor,
        incoming_gradient: &Tensor,
        outgoing_gradient: &mut Tensor,
    ) -> Result<(), &'static str> {
        if self.run_state == NetworkRunState::Inference {
            return Err("Can only do backprop in training mode");
        }
        // softmax Transoped @ incoming gradient -> yields 1x1x1 tensors
        Tensor::matrix_multiply_transpose_first(
            output,
            incoming_gradient,
            &mut self.intermediate1,
        )?;

        // Invert
        Tensor::scale_self(&mut self.intermediate1, -1.0);

        // Multiply with output

        Tensor::matrix_multiply(output, &self.intermediate1, &mut self.intermediate2);

        // Element wise multiplication for diagonal elements
        Tensor::multiply_elementwise(&output, incoming_gradient, outgoing_gradient);

        // Adding up both results

        Tensor::add_to_self(outgoing_gradient, &self.intermediate2);

        return Ok(());
    }
    fn get_output_shape(self: &Self) -> TensorShape {
        return self.shape.clone();
    }

    fn get_input_shape(self: &Self) -> TensorShape {
        return self.shape.clone();
    }

    fn get_name(self: &Self) -> String {
        return self.name.clone();
    }
    fn get_serialized(self: &Self) -> SerializedLayer {
        // Create empty map
        let mut serial_data: HashMap<SoftMaxSerialKeys, Vec<u8>> = HashMap::new();
        // Serialize elements that need to be serialized
        let serial_shape = bincode::serialize(&self.shape).unwrap();
        let serial_name = bincode::serialize(&self.name).unwrap();
        // Add to hashmap
        serial_data.insert(SoftMaxSerialKeys::Shape, serial_shape);
        serial_data.insert(SoftMaxSerialKeys::Name, serial_name);
        // Return wrapped in softmax layer
        return SerializedLayer::SerialSoftMaxLayer(serial_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: &SerializedLayer,
    ) -> Result<(), &'static str> {
        // Unwrapping the serialized layer and checking it is the correct type
        match serial_data {
            SerializedLayer::SerialSoftMaxLayer(data) => {
                return self.load_from_serialized_data(data);
            }
            _ => {
                return Err("Layer type does not match soft max type");
            }
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

    fn clear_gradients(self: &mut Self) {
        self.intermediate1.fill_with_value(0.0);
        self.intermediate2.fill_with_value(0.0);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let sml = SoftmaxLayer::new(
            TensorShape {
                di: 5,
                dj: 4,
                dk: 3,
            },
            &String::from("Softmax layer"),
        )
        .unwrap();
        let serialized_sml = sml.get_serialized();

        let mut sml2 =
            SoftmaxLayer::new(TensorShape::new(1, 1, 1), &String::from("Nothing")).unwrap();
        assert!(sml2.load_from_serialized(&serialized_sml).is_ok());

        assert_eq!(sml2.name, String::from("Softmax layer"));
        assert_eq!(
            sml2.shape,
            TensorShape {
                di: 5,
                dj: 4,
                dk: 3,
            },
        );
    }
}
