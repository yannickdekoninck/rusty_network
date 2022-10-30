use super::super::helper::{TensorIndex, TensorShape};
use super::super::Tensor;

pub fn check_convolution_dimensions(
    image_shape: &TensorShape,
    kernel_shape: &TensorShape,
    stride: u32,
    result_shape: &TensorShape,
    result_channel: u32,
) -> bool {
    // We check and immediately exit when something is not right
    // No need to check anything else

    // image and kernel depth must match
    if image_shape.dk != kernel_shape.dk {
        return false;
    }

    // Check if the kernel fitst in the image given the stride
    // We assume that the image is properly padded
    if (image_shape.di - kernel_shape.di) % stride != 0 {
        return false;
    }
    if (image_shape.dj - kernel_shape.dj) % stride != 0 {
        return false;
    }

    // Check if the output dimensions are correct
    let dim0_kernel_fits = (image_shape.di - kernel_shape.di) / stride + 1;
    let dim1_kernel_fits = (image_shape.dj - kernel_shape.dj) / stride + 1;
    if result_shape.di != dim0_kernel_fits {
        return false;
    }
    if result_shape.dj != dim1_kernel_fits {
        return false;
    }

    // check if the result channel is valid
    if result_channel >= result_shape.dk {
        return false;
    }

    return true;
}

pub fn convolution(
    image: &Tensor,
    kernel: &Tensor,
    stride: u32,
    result: &mut Tensor,
    result_channel: u32,
) -> Result<(), &'static str> {
    if !check_convolution_dimensions(
        &image.get_shape(),
        &kernel.get_shape(),
        stride,
        &result.get_shape(),
        result_channel,
    ) {
        return Err("Convolution dimensions do not match");
    }

    // All clear to convolute!

    let result_shape = result.get_shape();
    let kernel_shape = kernel.get_shape();

    // Main loop over image
    for j in 0..result_shape.dj {
        for i in 0..result_shape.di {
            let mut convolution_result: f32 = 0.0;
            let image_start_i = i * stride;
            let image_start_j = j * stride;

            // Loop over kernel dimensions and multiply - add
            for kk in 0..kernel_shape.dk {
                for kj in 0..kernel_shape.dj {
                    for ki in 0..kernel_shape.di {
                        let kernel_id = kernel.get_data_index(&TensorIndex {
                            i: ki,
                            j: kj,
                            k: kk,
                        }) as usize;
                        let image_id = image.get_data_index(&TensorIndex {
                            i: image_start_i + ki,
                            j: image_start_j + kj,
                            k: kk,
                        }) as usize;
                        convolution_result += image.data[image_id] * kernel.data[kernel_id];
                    }
                }
            }
            let result_id = result.get_data_index(&TensorIndex {
                i: i,
                j: j,
                k: result_channel,
            }) as usize;

            result.data[result_id] = convolution_result;
        }
    }
    return Ok(());
}

pub fn convolution_backprop_outgoing_gradient(
    incoming_gradients: &Tensor,
    kernels: &Vec<Tensor>,
    stride: u32,
    result: &mut Tensor,
) -> Result<(), &'static str> {
    // Check input shapes

    return Ok(());
}

pub fn convolution_backprop_kernel(
    incoming_gradients: &Tensor,
    image: &Tensor,
    stride: u32,
    kernel_channel: u32,
    kernel_gradient: &mut Tensor,
) -> Result<(), &'static str> {
    // Check input shapes
    if !check_convolution_dimensions(
        &image.get_shape(),
        &kernel_gradient.get_shape(),
        stride,
        &incoming_gradients.get_shape(),
        kernel_channel,
    ) {
        return Err("Convolution dimensions do not match");
    }

    // All clear to calculate!

    let result_shape = incoming_gradients.get_shape();
    let kernel_shape = kernel_gradient.get_shape();

    // Main loop over image
    for n in 0..kernel_shape.dk {
        for m in 0..kernel_shape.dj {
            for l in 0..kernel_shape.di {
                let mut convolution_result: f32 = 0.0;

                // Loop over incoming gradient dimensions and multiply - add
                for k in 0..result_shape.dk {
                    for j in 0..result_shape.dj {
                        for i in 0..result_shape.di {
                            let incoming_gradient_id = incoming_gradients
                                .get_data_index(&TensorIndex { i: i, j: j, k: k })
                                as usize;
                            let image_id = image.get_data_index(&TensorIndex {
                                i: l + i * stride,
                                j: m + j * stride,
                                k: n,
                            }) as usize;
                            convolution_result += image.data[image_id]
                                * incoming_gradients.data[incoming_gradient_id];
                        }
                    }
                }
                let kernel_gradient_id =
                    kernel_gradient.get_data_index(&TensorIndex { i: l, j: m, k: n }) as usize;

                kernel_gradient.data[kernel_gradient_id] = convolution_result;
            }
        }
    }
    return Ok(());
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::helper::TensorStride;
    #[test]
    fn test_convolution() {
        let shape_input = TensorShape::new(4, 4, 1);
        let shape_kernel = TensorShape::new(3, 3, 1);
        let shape_result = TensorShape::new(2, 2, 1);
        let stride: u32 = 1;
        let result_channel: u32 = 0;
        let mut tensor_image = Tensor::new(shape_input);
        assert!(tensor_image
            .fill_with_vec(vec![
                1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5,
            ])
            .is_ok());
        let tensor_kernel = Tensor {
            strides: TensorStride::new_from_shape(&shape_kernel),
            shape: shape_kernel,
            data: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let mut tensor_result = Tensor::new(shape_result);
        let tensor_expected_result = Tensor {
            strides: TensorStride::new_from_shape(&shape_result),
            shape: shape_result,
            data: vec![3.0, 2.0, 0.0, 2.5],
        };
        assert!(convolution(
            &tensor_image,
            &tensor_kernel,
            stride,
            &mut tensor_result,
            result_channel,
        )
        .is_ok());
        assert_eq!(tensor_result, tensor_expected_result);
    }

    #[test]
    fn test_convolution_backprop_kernel() {
        let shape_input = TensorShape::new(4, 4, 1);
        let shape_kernel = TensorShape::new(2, 2, 1);
        let shape_result = TensorShape::new(2, 2, 1);
        let stride: u32 = 2;
        let result_channel: u32 = 0;
        let mut tensor_image = Tensor::new(shape_input);
        assert!(tensor_image
            .fill_with_vec(vec![
                1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5,
            ])
            .is_ok());
        let mut tensor_incoming_gradient = Tensor::new(shape_result);
        tensor_incoming_gradient
            .fill_with_vec(vec![1.0, 2.0, 2.0, 0.0])
            .expect("Could not fill incoming gradient tensor");
        let mut tensor_kernel_gradient = Tensor::new(shape_kernel);
        let mut tensor_expected_kernel_gradient = Tensor::new(shape_kernel);
        tensor_expected_kernel_gradient
            .fill_with_vec(vec![1.0, 1.0, 1.0, 1.0])
            .expect("Could not fill incoming gradient tensor");
        assert!(convolution_backprop_kernel(
            &tensor_incoming_gradient,
            &tensor_image,
            stride,
            result_channel,
            &mut tensor_kernel_gradient,
        )
        .is_ok());
        assert_eq!(tensor_kernel_gradient, tensor_expected_kernel_gradient);
    }

    #[test]
    fn test_convolution_check_dimensions() {
        let shape_image = TensorShape::new(17, 25, 7);
        let shape_kernel = TensorShape::new(3, 3, 7);
        let shape_result = TensorShape::new(8, 12, 3);
        let stride: u32 = 2;
        let result_channel: u32 = 1;

        assert!(check_convolution_dimensions(
            &shape_image,
            &shape_kernel,
            stride,
            &shape_result,
            result_channel
        ));
        assert!(!check_convolution_dimensions(
            &shape_image,
            &shape_kernel,
            stride + 1,
            &shape_result,
            result_channel
        ));
    }
}
