use crate::layer::Layer;
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
