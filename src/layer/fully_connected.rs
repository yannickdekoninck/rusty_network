use super::SerializedLayer;
use crate::layer::Layer;
use crate::tensor::helper::TensorIndex;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;
use std::collections::HashMap;
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
        let serialized_data = HashMap::new();
        return SerializedLayer::SerialMaxPoolLayer(serialized_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: SerializedLayer,
    ) -> Result<(), &'static str> {
        return Ok(());
    }
}

#[cfg(test)]
mod test {
    use super::*;
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
}
