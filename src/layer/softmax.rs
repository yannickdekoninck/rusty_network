use crate::layer::Layer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;

pub struct SoftmaxLayer {
    shape: TensorShape,
    name: String,
}

impl SoftmaxLayer {
    pub fn new(shape: TensorShape, name: &String) -> Result<SoftmaxLayer, &'static str> {
        let cl = SoftmaxLayer {
            shape: shape,
            name: name.clone(),
        };

        return Ok(cl);
    }
}

impl Layer for SoftmaxLayer {
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor) {
        Tensor::softmax(input, output);
        return;
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
}
