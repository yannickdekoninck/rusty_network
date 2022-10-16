use super::SerializedLayer;
use crate::layer::Layer;

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
    bias: Tensor,
    name: String,
}

impl FullyConnectedLayer {
    pub fn new(input_size: u32, output_size: u32, name: &String) -> FullyConnectedLayer {
        let weights = Tensor::new(TensorShape::new(output_size, input_size, 1));
        let bias: Tensor = Tensor::new(TensorShape {
            di: output_size,
            dj: 1,
            dk: 1,
        });
        return FullyConnectedLayer {
            weights: weights,
            bias: bias,
            name: name.clone(),
        };
    }

    pub fn empty() -> Self {
        let fc = FullyConnectedLayer {
            weights: Tensor::new(TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            }),
            bias: Tensor::new(TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            }),
            name: String::from("Empty fully connected layer"),
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
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor) {
        Tensor::matrix_multiply_add_relu(input, &self.weights, &self.bias, output);
        return;
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
}
