use crate::layer::Layer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;

pub struct ConvolutionalLayer {
    kernels: Vec<Tensor>,
    stride: u32,
    input_shape: TensorShape,
    output_shape: TensorShape,
    name: String,
}

impl ConvolutionalLayer {
    pub fn new(
        kernel_shape: TensorShape,
        input_shape: TensorShape,
        stride: u32,
        output_depth: u32,
        name: &String,
    ) -> Result<ConvolutionalLayer, &'static str> {
        if !Tensor::does_kernel_stride_fit_image(&input_shape, &kernel_shape, stride) {
            return Err("Kernel does not fit in image with given stride");
        }
        let (dim0, dim1) = Tensor::get_convolution_dim_fit(&input_shape, &kernel_shape, stride);

        let result_shape = TensorShape {
            di: dim0,
            dj: dim1,
            dk: output_depth,
        };

        let kernels = vec![Tensor::new(kernel_shape); output_depth as usize];

        let cl = ConvolutionalLayer {
            kernels: kernels,
            stride: stride,
            input_shape: input_shape,
            output_shape: result_shape,
            name: name.clone(),
        };

        return Ok(cl);
    }
}

impl Layer for ConvolutionalLayer {
    fn forward(self: &Self, input: &Tensor, output: &mut Tensor) {
        // Convolution for every output channel
        for i in 0..self.output_shape.dk {
            Tensor::convolution(input, &self.kernels[i as usize], self.stride, output, i);
        }
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
