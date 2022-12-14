use serde::Deserialize;
use serde::Serialize;

use std::fs;

use crate::layer::Layer;
use crate::layer::SerializedLayer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;
use std::error::Error;

// An enum to indicate the state of a network and its layers

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NetworkRunState {
    Inference,
    Training,
}

// This datastructure is used to store the network topology and data
#[derive(Serialize, Deserialize)]
struct SerializedNetwork {
    serial_layers: Vec<SerializedLayer>,
}

impl SerializedNetwork {
    fn empty() -> SerializedNetwork {
        let snl = SerializedNetwork {
            serial_layers: vec![],
        };
        return snl;
    }

    fn populate_network(self: &Self, network: &mut Network) -> Result<(), &'static str> {
        for layer in &self.serial_layers {
            SerializedLayer::add_to_network(layer, network)?;
        }
        return Ok(());
    }

    fn new_from_network(network: &Network) -> Result<SerializedNetwork, &'static str> {
        let mut sn = SerializedNetwork::empty();
        let layers = network.get_layers();
        for layer in layers {
            let serial_layer = layer.get_serialized();
            sn.serial_layers.push(serial_layer);
        }
        return Ok(sn);
    }

    fn to_byte_array(self: &Self) -> Result<Vec<u8>, &'static str> {
        return match bincode::serialize(self) {
            Ok(val) => Ok(val),
            Err(_) => Err("Could not convert to by array"),
        };
    }
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    intermediate_states: Vec<Tensor>,
    intermediate_gradients: Vec<Tensor>,
    input_shape: TensorShape,
    output_shape: TensorShape,
    run_state: NetworkRunState,
}

impl Network {
    // Create a new network
    pub fn new(run_state: NetworkRunState) -> Network {
        return Network {
            layers: vec![],
            input_shape: TensorShape {
                di: 0,
                dj: 0,
                dk: 0,
            },
            output_shape: TensorShape {
                di: 0,
                dj: 0,
                dk: 0,
            },
            intermediate_states: vec![],
            intermediate_gradients: vec![],
            run_state: run_state,
        };
    }

    pub fn save_to_file(self: &Self, filename: &String) -> Result<(), Box<dyn Error>> {
        let serialized_network = SerializedNetwork::new_from_network(self)?;
        let byte_stream = serialized_network.to_byte_array()?;
        fs::write(filename, byte_stream)?;
        return Ok(());
    }
    pub fn load_from_file(self: &mut Self, filename: &String) -> Result<(), Box<dyn Error>> {
        let file_content = fs::read(filename)?;
        let sn: SerializedNetwork = bincode::deserialize(&file_content)?;
        sn.populate_network(self)?;
        return Ok(());
    }

    pub fn add_layer<T: Layer + 'static>(
        self: &mut Self,
        mut layer: T,
    ) -> Result<(), &'static str> {
        // Make sure the new layer is in the correct state
        match self.run_state {
            NetworkRunState::Inference => {
                layer.switch_to_inference();
            }
            NetworkRunState::Training => {
                layer.switch_to_training();
            }
        }

        // Check if this is the first layer to be added
        if self.layers.len() == 0 {
            // This is the first layer
            // Copy over the interface shapes of this layer
            self.input_shape = layer.get_input_shape();
            self.output_shape = layer.get_output_shape();
            // Push layer to list
            self.layers.push(Box::new(layer));
            // Adding input and outputs as new intermediate states
            self.intermediate_states
                .push(Tensor::new(self.input_shape.clone()));
            self.intermediate_states
                .push(Tensor::new(self.output_shape.clone()));
            if self.run_state == NetworkRunState::Training {
                // Adding input and outputs as new intermediate states
                self.intermediate_gradients
                    .push(Tensor::new(self.input_shape.clone()));
                self.intermediate_gradients
                    .push(Tensor::new(self.output_shape.clone()));
            }
            return Ok(());
        }

        // Check if this new layer matches the existing network
        let last_layer_shape = self.layers.last().unwrap().get_output_shape();
        if last_layer_shape != layer.get_input_shape() {
            return Err("Last output and new input shapes don't match");
        }

        // Set the output shape
        self.output_shape = layer.get_output_shape();
        // Push layer to list
        self.layers.push(Box::new(layer));
        // Add a new intermediate state
        self.intermediate_states
            .push(Tensor::new(self.output_shape.clone()));

        if self.run_state == NetworkRunState::Training {
            self.intermediate_gradients
                .push(Tensor::new(self.output_shape.clone()));
        }

        return Ok(());
    }

    // Input shape getter
    pub fn get_input_shape(self: &Self) -> TensorShape {
        return self.input_shape.clone();
    }

    // Output shape getter
    pub fn get_output_shape(self: &Self) -> TensorShape {
        return self.output_shape.clone();
    }

    // Intermediate tensor getter
    pub fn get_intermediate_state(self: &Self, id: usize) -> Result<&Tensor, &'static str> {
        if id >= self.intermediate_states.len() {
            return Err("Trying to access an intermediate state that does not exist");
        }
        return Ok(&self.intermediate_states[id]);
    }

    pub fn get_layers(self: &Self) -> &Vec<Box<dyn Layer>> {
        return &self.layers;
    }

    pub fn get_run_state(self: &Self) -> NetworkRunState {
        return self.run_state.clone();
    }
    pub fn switch_to_training(&mut self) {
        self.run_state = NetworkRunState::Training;
        for layer in self.layers.iter_mut() {
            layer.switch_to_training();
        }
        self.intermediate_gradients = vec![];
        for int_state in self.intermediate_states.iter() {
            self.intermediate_gradients
                .push(Tensor::new(int_state.get_shape()));
        }
    }
    pub fn switch_to_inference(&mut self) {
        self.run_state = NetworkRunState::Training;
        for layer in self.layers.iter_mut() {
            layer.switch_to_inference();
        }
        // No need to keep track of intermediate gradients
        self.intermediate_gradients = vec![];
    }

    pub fn infer(self: &mut Self, input: &Tensor) -> Result<(), &'static str> {
        // Input shape check
        if input.get_shape() != self.input_shape {
            return Err("Input shape not as expected");
        }

        if self.intermediate_states.len() < 2 {
            return Err("Network does not contain any layers");
        }

        // Copy the input tensor into the intermediate states vector
        self.intermediate_states[0] = input.clone();

        // Create a slice to index into
        let intermediate_state_slice = &mut self.intermediate_states[..];

        for i in 0..self.layers.len() {
            // Split up the slice into two mutable slices
            // This is required to be able to both borrow the output mutably and the input immutably
            let (input_slice, output_slice) = intermediate_state_slice.split_at_mut(i + 1);

            // Load the layer
            let layer = &mut self.layers[i];

            // Get the correct input and output
            let input_tensor = input_slice.last().unwrap();
            let output_tensor = output_slice.first_mut().unwrap();

            // Propagate through network
            layer.forward(input_tensor, output_tensor)?;
        }

        return Ok(());
    }

    pub fn backward(self: &mut Self, incoming_loss_gradient: &Tensor) -> Result<(), &'static str> {
        // Input shape check
        if incoming_loss_gradient.get_shape() != self.output_shape {
            return Err("Incoming loss gradient shape not as expected");
        }

        // Check there are layers
        if self.layers.len() < 1 {
            return Err("Network does not contain any layers");
        }

        // Check we are in the correct mode
        if self.run_state != NetworkRunState::Training {
            return Err("Can only propagate backwards in training mode");
        }

        let number_of_layers = self.layers.len();
        // Copy the incoming loss gradient into the intermediate states vector
        self.intermediate_gradients[number_of_layers] = incoming_loss_gradient.clone();

        // Create a slice to index into
        let intermediate_gradients_slice = &mut self.intermediate_gradients[..];

        for i in (0..self.layers.len()).rev() {
            // Split up the slice into two mutable slices
            // This is required to be able to both borrow the output mutably and the input immutably
            let (outgoing_slice, incoming_slice) = intermediate_gradients_slice.split_at_mut(i + 1);

            // Load the layer
            let layer = &mut self.layers[i];

            // Get the correct input and output
            let incoming_gradient = incoming_slice.first().unwrap();
            let outgoing_gradient = outgoing_slice.last_mut().unwrap();
            let input = &self.intermediate_states[i];
            let output = &self.intermediate_states[i + 1];

            // Propagate through network
            layer.backward(input, output, incoming_gradient, outgoing_gradient)?;
        }

        return Ok(());
    }

    pub fn get_output(self: &Self) -> Result<&Tensor, &'static str> {
        if self.intermediate_states.len() < 2 {
            return Err("Network does not contain an output");
        }
        return Ok(&self.intermediate_states.last().unwrap());
    }
}

#[cfg(test)]
mod test {

    use crate::{
        layer::{convolutional::ConvolutionalLayer, fully_connected::FullyConnectedLayer},
        tensor::helper::TensorShape,
    };

    use super::*;
    #[test]
    fn test_adding_layer() {
        let layer_1 = ConvolutionalLayer::new(
            TensorShape::new(2, 2, 1),
            TensorShape::new(4, 2, 1),
            1,
            1,
            &String::from("conv_layer_1"),
        )
        .unwrap();

        let layer_2 = FullyConnectedLayer::new(3, 8, &String::from("full_layer_1"));
        let layer_3 = FullyConnectedLayer::new(3, 8, &String::from("full_layer_2"));
        let expected_output_shape = layer_2.get_output_shape();
        let expected_input_shape = layer_1.get_input_shape();
        let mut network = Network::new(NetworkRunState::Inference);
        assert!(network.add_layer(layer_1).is_ok());
        assert!(network.add_layer(layer_2).is_ok());
        assert!(network.add_layer(layer_3).is_err());
        let input_shape = network.get_input_shape();
        let output_shape = network.get_output_shape();
        assert_eq!(input_shape, expected_input_shape);
        assert_eq!(output_shape, expected_output_shape);
    }

    #[test]
    fn test_run_state() {
        let layer_1 = FullyConnectedLayer::new(3, 8, &String::from("full_layer_1"));
        let layer_2 = FullyConnectedLayer::new(8, 4, &String::from("full_layer_2"));
        let mut network = Network::new(NetworkRunState::Inference);
        assert!(network.add_layer(layer_1).is_ok());

        for lay in network.layers.iter() {
            assert_eq!(lay.get_run_mode(), NetworkRunState::Inference);
        }
        network.switch_to_training();
        for lay in network.layers.iter() {
            assert_eq!(lay.get_run_mode(), NetworkRunState::Training);
        }
        assert!(network.add_layer(layer_2).is_ok());
        for lay in network.layers.iter() {
            assert_eq!(lay.get_run_mode(), NetworkRunState::Training);
        }
        network.switch_to_inference();
        for lay in network.layers.iter() {
            assert_eq!(lay.get_run_mode(), NetworkRunState::Inference);
        }
    }

    #[test]
    fn test_getting_intermediate_states() {
        let layer_1 = ConvolutionalLayer::new(
            TensorShape::new(2, 2, 1),
            TensorShape::new(4, 2, 1),
            1,
            1,
            &String::from("conv_layer_1"),
        )
        .unwrap();

        let layer_2 = FullyConnectedLayer::new(3, 8, &String::from("full_layer_1"));
        let expected_intermediate_shape = layer_1.get_output_shape();
        let mut network = Network::new(NetworkRunState::Inference);
        assert!(network.add_layer(layer_1).is_ok());
        assert!(network.add_layer(layer_2).is_ok());
        let intermediate_shape = network.get_intermediate_state(1).unwrap().get_shape();
        assert_eq!(intermediate_shape, expected_intermediate_shape);
    }
    #[test]
    fn test_network_infer() {
        // Create layers
        let mut layer_1 = FullyConnectedLayer::new(2, 3, &String::from("full_layer_1"));
        let mut layer_2 = FullyConnectedLayer::new(3, 2, &String::from("full_layer_2"));

        // Fill weights and bias values
        layer_1.fill_weights_with_value(1.0);
        layer_1.fill_bias_with_value(2.0);

        layer_2.fill_weights_with_value(3.0);
        layer_2.fill_bias_with_value(4.0);

        // Create input
        let mut input_tensor = Tensor::new(TensorShape::new(2, 1, 1));
        input_tensor.fill_with_value(1.0);

        // Create network
        let mut network = Network::new(NetworkRunState::Inference);
        network.add_layer(layer_1).unwrap();
        network.add_layer(layer_2).unwrap();

        // Infer and check this does not thrown an error
        assert!(network.infer(&input_tensor).is_ok());

        // Construct the expected result tensor
        let result = network.get_output().unwrap();

        let mut expected_result = Tensor::new(TensorShape {
            di: 2,
            dj: 1,
            dk: 1,
        });
        expected_result.fill_with_value(40.0);

        // Check the result is as expected
        assert_eq!(expected_result, *result);
    }

    #[test]
    fn test_network_backward() {
        // Create layers
        let mut layer_1 = FullyConnectedLayer::new(2, 3, &String::from("full_layer_1"));
        let mut layer_2 = FullyConnectedLayer::new(3, 2, &String::from("full_layer_2"));

        // Fill weights and bias values
        layer_1.fill_weights_with_value(1.0);
        layer_1.fill_bias_with_value(2.0);

        layer_2.fill_weights_with_value(3.0);
        layer_2.fill_bias_with_value(4.0);

        // Create input
        let mut input_tensor = Tensor::new(TensorShape::new(2, 1, 1));
        input_tensor.fill_with_value(1.0);

        // Create network
        let mut network = Network::new(NetworkRunState::Inference);
        network.add_layer(layer_1).unwrap();
        network.add_layer(layer_2).unwrap();

        network.switch_to_training();

        // Infer and check this does not thrown an error
        assert!(network.infer(&input_tensor).is_ok());

        // Construct the expected result tensor
        let result = network.get_output().unwrap();

        let mut expected_result = Tensor::new(TensorShape {
            di: 2,
            dj: 1,
            dk: 1,
        });
        expected_result.fill_with_value(40.0);

        // Check the result is as expected
        assert_eq!(expected_result, *result);

        let mut incoming_gradient = Tensor::new(result.get_shape());
        incoming_gradient.fill_with_value(1.0);

        // Backprop
        assert!(network.backward(&incoming_gradient).is_ok());

        // Expected tensors
        let mut expected_intermediate_gradient = Tensor::new(TensorShape::new_1d(3));
        expected_intermediate_gradient.fill_with_value(6.0);
        let mut expected_final_gradient = Tensor::new(input_tensor.get_shape());
        expected_final_gradient.fill_with_value(18.0);

        assert_eq!(
            expected_intermediate_gradient,
            *network.intermediate_gradients.get(1).unwrap()
        );
        assert_eq!(
            expected_final_gradient,
            *network.intermediate_gradients.first().unwrap()
        );
    }

    #[test]
    fn test_network_serialize_deserialize() {
        // Create layers
        let mut layer_1 = FullyConnectedLayer::new(2, 3, &String::from("full_layer_1"));
        let mut layer_2 = FullyConnectedLayer::new(3, 2, &String::from("full_layer_2"));

        // Fill weights and bias values
        layer_1.fill_weights_with_value(1.0);
        layer_1.fill_bias_with_value(2.0);

        layer_2.fill_weights_with_value(3.0);
        layer_2.fill_bias_with_value(4.0);

        // Create network
        let mut network = Network::new(NetworkRunState::Inference);
        network.add_layer(layer_1).unwrap();
        network.add_layer(layer_2).unwrap();

        // Create input
        let mut input_tensor = Tensor::new(TensorShape::new(2, 1, 1));
        input_tensor.fill_with_value(1.0);

        network.infer(&input_tensor).unwrap();
        let output = network.get_output().unwrap();

        let serialized_network = SerializedNetwork::new_from_network(&network).unwrap();

        let mut deserialized_network = Network::new(NetworkRunState::Inference);

        assert!(serialized_network
            .populate_network(&mut deserialized_network)
            .is_ok());

        assert_eq!(deserialized_network.layers.len(), 2);
        let nl1 = deserialized_network.layers[0].get_name();
        let ol1 = network.layers[0].get_name();
        assert_eq!(nl1, ol1);

        assert!(deserialized_network.infer(&input_tensor).is_ok());
        let other_output = deserialized_network.get_output().unwrap();

        assert_eq!(output, other_output);
    }
}
