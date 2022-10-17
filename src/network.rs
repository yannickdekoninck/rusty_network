use serde::Deserialize;
use serde::Serialize;

use std::fs;

use crate::layer::Layer;
use crate::layer::SerializedLayer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;
use std::error::Error;

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
    input_shape: TensorShape,
    output_shape: TensorShape,
}

impl Network {
    // Create a new network
    pub fn new() -> Network {
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

    pub fn add_layer<T: Layer + 'static>(self: &mut Self, layer: T) -> Result<(), &'static str> {
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
            let layer = &self.layers[i];

            // Get the correct input and output
            let input_tensor = input_slice.last().unwrap();
            let output_tensor = output_slice.first_mut().unwrap();

            // Propagate through network
            layer.forward(input_tensor, output_tensor);
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
        let mut network = Network::new();
        assert!(network.add_layer(layer_1).is_ok());
        assert!(network.add_layer(layer_2).is_ok());
        assert!(network.add_layer(layer_3).is_err());
        let input_shape = network.get_input_shape();
        let output_shape = network.get_output_shape();
        assert_eq!(input_shape, expected_input_shape);
        assert_eq!(output_shape, expected_output_shape);
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
        let mut network = Network::new();
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
        let mut network = Network::new();
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
        let mut network = Network::new();
        network.add_layer(layer_1).unwrap();
        network.add_layer(layer_2).unwrap();

        // Create input
        let mut input_tensor = Tensor::new(TensorShape::new(2, 1, 1));
        input_tensor.fill_with_value(1.0);

        network.infer(&input_tensor).unwrap();
        let output = network.get_output().unwrap();

        let serialized_network = SerializedNetwork::new_from_network(&network).unwrap();

        let mut deserialized_network = Network::new();

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
