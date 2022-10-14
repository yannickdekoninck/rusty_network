use crate::layer::Layer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;

pub struct MaxPoolLayer {
    input_shape: TensorShape,
    mask_shape: TensorShape,
    output_shape: TensorShape,
    stride: u32,
    name: String,
}

impl MaxPoolLayer {
    pub fn new(
        mask_shape: TensorShape,
        input_shape: TensorShape,
        stride: u32,
        name: &String,
    ) -> Result<MaxPoolLayer, &'static str> {
        if !Tensor::does_kernel_stride_fit_image(&input_shape, &mask_shape, stride) {
            return Err("Mask does not fit in image with given stride");
        }
        let (dim0, dim1) = Tensor::get_convolution_dim_fit(&input_shape, &mask_shape, stride);

        let result_shape = TensorShape {
            di: dim0,
            dj: dim1,
            dk: input_shape.dk,
        };

        let cl = MaxPoolLayer {
            input_shape: input_shape,
            stride: stride,
            mask_shape: mask_shape,
            output_shape: result_shape,
            name: name.clone(),
        };

        return Ok(cl);
    }
}

impl Layer for MaxPoolLayer {
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor) {
        Tensor::max_pool(input, &self.mask_shape, self.stride, output);
        return;
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
}
