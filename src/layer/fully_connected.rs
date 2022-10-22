use super::SerializedLayer;
use crate::{layer::Layer, network::NetworkRunState};

use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum FullyConnectedSerialKeys {
    Weights,
    Bias,
    Name,
}

pub struct FullyConnectedLayer {
    weights: Tensor,
    weights_gradients: Tensor,
    weights_gradients_intermediate: Tensor,
    bias: Tensor,
    bias_gradients: Tensor,
    relu_mask: Tensor,
    name: String,
    run_state: NetworkRunState,
}

impl FullyConnectedLayer {
    fn update_gradient_and_intermediate_tensors(self: &mut Self) {
        match self.run_state {
            NetworkRunState::Inference => {
                // Clean up all training related tensors
                self.weights_gradients = Tensor::empty();
                self.weights_gradients_intermediate = Tensor::empty();
                self.bias_gradients = Tensor::empty();
                self.relu_mask = Tensor::empty();
            }
            NetworkRunState::Training => {
                // Allocate all tensors required for training
                self.weights_gradients = Tensor::new(self.weights.get_shape());
                self.weights_gradients_intermediate = Tensor::new(self.weights.get_shape());
                self.bias_gradients = Tensor::new(self.bias.get_shape());
                self.relu_mask = Tensor::new(self.get_output_shape());
                // Fill up all tensors with 0.0
                self.clear_gradients();
            }
        }
    }

    pub fn new(input_size: u32, output_size: u32, name: &String) -> FullyConnectedLayer {
        let mut fcl = FullyConnectedLayer::empty();
        let weights = Tensor::new(TensorShape::new(output_size, input_size, 1));
        let bias: Tensor = Tensor::new(TensorShape {
            di: output_size,
            dj: 1,
            dk: 1,
        });
        fcl.fill_from_state(weights, bias, name).unwrap();
        return fcl;
    }

    pub fn empty() -> Self {
        // Initialize an empty fully connected layer
        let fc = FullyConnectedLayer {
            weights: Tensor::empty(),
            bias: Tensor::empty(),
            weights_gradients: Tensor::empty(),
            weights_gradients_intermediate: Tensor::empty(),
            bias_gradients: Tensor::empty(),
            relu_mask: Tensor::empty(),
            name: String::from("Empty fully connected layer"),
            run_state: NetworkRunState::Inference,
        };

        return fc;
    }

    pub fn fill_from_state(
        self: &mut Self,
        weights: Tensor,
        bias: Tensor,
        name: &String,
    ) -> Result<(), &'static str> {
        // Check sizes
        let weight_shape = weights.get_shape();
        let bias_shape = bias.get_shape();
        if weight_shape.dk != bias_shape.dk {
            return Err("Third dimension of bias and weights don't match");
        }
        if weight_shape.di != bias_shape.di {
            return Err("First dimension of bias and weight don't match");
        }
        if bias_shape.dj != 1 {
            return Err("The second dimension of bias must be 1");
        }

        //ready to assign
        self.weights = weights;
        self.bias = bias;
        self.name = name.clone();
        // Update the helper tensors
        self.update_gradient_and_intermediate_tensors();
        return Ok(());
    }

    pub fn fill_weights_with_value(self: &mut Self, value: f32) {
        self.weights.fill_with_value(value);
    }

    pub fn fill_bias_with_value(self: &mut Self, value: f32) {
        self.bias.fill_with_value(value);
    }

    fn load_from_serialized_data(
        self: &mut Self,
        data: &HashMap<FullyConnectedSerialKeys, Vec<u8>>,
    ) -> Result<(), &'static str> {
        // Check the correct keys are present
        if !data.contains_key(&FullyConnectedSerialKeys::Name) {
            return Err("Data does not contain --name-- key");
        }
        if !data.contains_key(&FullyConnectedSerialKeys::Weights) {
            return Err("Data does not contain --weights-- key");
        }
        if !data.contains_key(&FullyConnectedSerialKeys::Bias) {
            return Err("Data does not contain --bias-- key");
        }

        let weights_data = data
            .get(&FullyConnectedSerialKeys::Weights)
            .expect("Cannot access weigths shape data");
        let weights: Tensor =
            bincode::deserialize(weights_data).expect("Cannot deserialize weights data");

        let bias_data = data
            .get(&FullyConnectedSerialKeys::Bias)
            .expect("Cannot access input shape data");
        let bias: Tensor = bincode::deserialize(bias_data).expect("Cannot deserialize bias data");

        let name_data = data
            .get(&FullyConnectedSerialKeys::Name)
            .expect("Cannot access name data");
        let name: String = bincode::deserialize(name_data).expect("Cannot deserialize name data");

        return self.fill_from_state(weights, bias, &name);
    }
}

impl Layer for FullyConnectedLayer {
    fn forward(self: &mut Self, input: &Tensor, output: &mut Tensor) {
        match self.run_state {
            NetworkRunState::Inference => {
                // Take the faster multiply add relu in 1 go
                Tensor::matrix_multiply_add_relu(input, &self.weights, &self.bias, output);
            }
            NetworkRunState::Training => {
                // Take the piecewise route so we can keep track of the relu mask which we need for backprop

                // Multiply with weights
                Tensor::matrix_multiply(&self.weights, input, output);
                // Add bias
                Tensor::add_to_self(output, &self.bias);
                // ReLu
                Tensor::relu_self_and_store_mask(output, &mut self.relu_mask).unwrap();
            }
        }
        return;
    }

    fn backward(
        self: &mut Self,
        input: &Tensor,
        incoming_gradient: &Tensor,
        outgoing_gradient: &mut Tensor,
    ) {
        if self.run_state == NetworkRunState::Training {
            // bias gradient
            Tensor::add_to_self(&mut self.bias_gradients, incoming_gradient);
            // weights gradient
            Tensor::matrix_multiply_transpose_second(
                incoming_gradient,
                input,
                &mut self.weights_gradients_intermediate,
            )
            .unwrap();
        }
    }

    fn get_output_shape(self: &Self) -> TensorShape {
        return self.bias.get_shape();
    }

    fn get_input_shape(self: &Self) -> TensorShape {
        return TensorShape {
            di: self.weights.get_shape().dj,
            dj: 1,
            dk: 1,
        };
    }

    fn get_name(self: &Self) -> String {
        return self.name.clone();
    }

    fn get_run_mode(self: &Self) -> NetworkRunState {
        return self.run_state.clone();
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

    fn clear_gradients(self: &mut Self) {
        self.weights_gradients_intermediate.fill_with_value(0.0);
        self.weights_gradients.fill_with_value(0.0);
        self.bias_gradients.fill_with_value(0.0);
        self.relu_mask.fill_with_value(0.0);
    }

    fn get_serialized(self: &Self) -> SerializedLayer {
        // Create empty map
        let mut serial_data: HashMap<FullyConnectedSerialKeys, Vec<u8>> = HashMap::new();
        // Serialize elements that need to be serialized
        let serial_weights = bincode::serialize(&self.weights).unwrap();
        let serial_bias = bincode::serialize(&self.bias).unwrap();
        let serial_name = bincode::serialize(&self.name).unwrap();
        // Add to hashmap
        serial_data.insert(FullyConnectedSerialKeys::Weights, serial_weights);
        serial_data.insert(FullyConnectedSerialKeys::Bias, serial_bias);
        serial_data.insert(FullyConnectedSerialKeys::Name, serial_name);
        // Return wrapped in softmax layer
        return SerializedLayer::SerialFullyConnectedLayer(serial_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: &SerializedLayer,
    ) -> Result<(), &'static str> {
        // Unwrapping the serialized layer and checking it is the correct type
        match serial_data {
            SerializedLayer::SerialFullyConnectedLayer(data) => {
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
    use crate::tensor::helper::TensorIndex;
    #[test]
    fn test_value_filling() {
        let mut fc_layer = FullyConnectedLayer::new(4, 4, &String::from("Test layer"));
        fc_layer.fill_weights_with_value(7.0);
        fc_layer.fill_bias_with_value(25.0);

        let bias_probe = fc_layer.bias.get_item(&TensorIndex::index_1d(2));
        let weight_probe = fc_layer.weights.get_item(&TensorIndex::index_2d(1, 2));

        assert_eq!(bias_probe, 25.0);
        assert_eq!(weight_probe, 7.0);
    }

    #[test]
    fn test_serialize_deserialize() {
        let mut fc_layer = FullyConnectedLayer::new(4, 4, &String::from("Test layer"));
        fc_layer.fill_weights_with_value(7.0);
        fc_layer.fill_bias_with_value(25.0);

        let mut other_layer = FullyConnectedLayer::empty();

        let serialized_layer = fc_layer.get_serialized();

        assert!(other_layer.load_from_serialized(&serialized_layer).is_ok());

        assert_eq!(other_layer.weights, fc_layer.weights);
        assert_eq!(other_layer.bias, fc_layer.bias);
        assert_eq!(other_layer.name, fc_layer.name);
    }

    #[test]
    fn test_backprop() {
        // Create layer

        let mut fcl = FullyConnectedLayer::empty();
        let mut weights = Tensor::new(TensorShape::new_2d(3, 3));
        assert!(weights
            .fill_with_vec(vec![1.0, 2.0, 3.0, -1.0, -2.0, -8.0, 1.0, 1.0, -2.0])
            .is_ok());

        let mut bias = Tensor::new(TensorShape::new_1d(3));
        assert!(bias.fill_with_vec(vec![1.0, 0.0, 2.0]).is_ok());

        let mut input = Tensor::new(TensorShape::new_1d(3));
        assert!(input.fill_with_vec(vec![2.0, 1.0, 1.0]).is_ok());

        let mut expected_output = Tensor::new(TensorShape::new_1d(3));
        assert!(expected_output.fill_with_vec(vec![3.0, 3.0, 0.0]).is_ok());

        let mut output = Tensor::new(TensorShape::new_1d(3));
        let mut outgoing_gradient = Tensor::new(TensorShape::new_1d(3));
        let mut incoming_gradient = Tensor::new(TensorShape::new_1d(3));

        let mut expected_outgoing_gradient = Tensor::new(TensorShape::new_1d(3));
        expected_outgoing_gradient.fill_with_vec(vec![5.0, -5.0, 3.0]);

        incoming_gradient.fill_with_vec(vec![1.0, 2.0, 3.0]);

        assert!(fcl
            .fill_from_state(weights, bias, &String::from("Test layer"))
            .is_ok());

        fcl.switch_to_learning();

        fcl.forward(&input, &mut output);

        fcl.backward(&input, &incoming_gradient, &mut outgoing_gradient);

        assert_eq!(output, expected_output);
    }
}
