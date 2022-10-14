use crate::layer::Layer;
use crate::tensor::helper::TensorIndex;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;

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
