use super::super::helper::{TensorIndex, TensorShape};
use super::super::Tensor;

pub fn check_convolution_dimensions(
    image_shape: &TensorShape,
    kernel_shape: &TensorShape,
    stride: u32,
    result_shape: &TensorShape,
    result_channel: u32,
) -> Result<(), &'static str> {
    // We check and immediately exit when something is not right
    // No need to check anything else

    // image and kernel depth must match
    if image_shape.dk != kernel_shape.dk {
        return Err("Third dimension of image and shpe don't match");
    }

    // Check if the kernel fitst in the image given the stride
    // We assume that the image is properly padded
    if (image_shape.di - kernel_shape.di) % stride != 0 {
        return Err("First dimensions of kernel and shape do not match with stride");
    }
    if (image_shape.dj - kernel_shape.dj) % stride != 0 {
        return Err("Secon dimensions of kernel and shape do not match with stride");
    }

    // Check if the output dimensions are correct
    let dim0_kernel_fits = (image_shape.di - kernel_shape.di) / stride + 1;
    let dim1_kernel_fits = (image_shape.dj - kernel_shape.dj) / stride + 1;
    if result_shape.di != dim0_kernel_fits {
        return Err("Results shape first dimension does not match with stride");
    }
    if result_shape.dj != dim1_kernel_fits {
        return Err("Results shape second dimension does not match with stride");
    }

    // check if the result channel is valid
    if result_channel >= result_shape.dk {
        return Err("Results channel is too high");
    }

    return Ok(());
}

pub fn convolution(
    image: &Tensor,
    kernel: &Tensor,
    stride: u32,
    result: &mut Tensor,
    result_channel: u32,
) -> Result<(), &'static str> {
    check_convolution_dimensions(
        &image.get_shape(),
        &kernel.get_shape(),
        stride,
        &result.get_shape(),
        result_channel,
    )?;

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
pub fn convolution_bias_relu(
    image: &Tensor,
    kernel: &Tensor,
    bias: &Tensor,
    stride: u32,
    result: &mut Tensor,
    result_channel: u32,
) -> Result<(), &'static str> {
    check_convolution_dimensions(
        &image.get_shape(),
        &kernel.get_shape(),
        stride,
        &result.get_shape(),
        result_channel,
    )?;

    let bias_shape = bias.get_shape();

    if bias_shape.di > 1 || bias_shape.dj > 1 || bias_shape.dk > 1 {
        return Err("bias should be a 1x1x1 tensor");
    }

    let bias_value = bias.get_item(&TensorIndex { i: 0, j: 0, k: 0 });

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

            // Add bias value
            convolution_result += bias_value;
            // cheap relu
            convolution_result = convolution_result.max(0.0);

            result.data[result_id] = convolution_result;
        }
    }
    return Ok(());
}

pub fn convolution_backprop_outgoing_gradient(
    incoming_gradients: &Tensor,
    kernels: &Vec<Tensor>,
    stride: u32,
    outgoing_gradients: &mut Tensor,
) -> Result<(), &'static str> {
    // Check input shapes for all kernels
    for (i, kernel) in kernels.iter().enumerate() {
        check_convolution_dimensions(
            &outgoing_gradients.get_shape(),
            &kernel.get_shape(),
            stride,
            &incoming_gradients.get_shape(),
            i as u32,
        )?;
    }

    let incoming_gradient_shape = incoming_gradients.get_shape();

    // Additional check for kernel vec length
    if kernels.len() as u32 != incoming_gradient_shape.dk {
        return Err("Number of kernels does not match output dimension dk");
    }

    // All clear to calculate!

    let outgoing_gradient_shape = outgoing_gradients.get_shape();

    // Main loop over outgoing gradient
    for c in 0..outgoing_gradient_shape.dk {
        for b in 0..outgoing_gradient_shape.dj {
            for a in 0..outgoing_gradient_shape.di {
                let mut convolution_result: f32 = 0.0;
                // Loop over incoming gradient dimensions and multiply - add
                for k in 0..incoming_gradient_shape.dk {
                    let kernel = &(kernels[k as usize]);
                    let kernel_shape = kernel.get_shape();
                    let i_start = (a as i32 - kernel_shape.di as i32) / stride as i32 + 1;
                    // Now converting to unsigned
                    let i_start = i_start.max(0) as u32;
                    let mut i_stop = a / stride + 1;
                    i_stop = i_stop.min(kernel_shape.di);
                    let j_start = (b as i32 - kernel_shape.dj as i32) / stride as i32 + 1;
                    let j_start = j_start.max(0) as u32;
                    let mut j_stop = b / stride + 1;
                    j_stop = j_stop.min(kernel_shape.dj);
                    for j in j_start..j_stop {
                        for i in i_start..i_stop {
                            let incoming_gradient_id = incoming_gradients
                                .get_data_index(&TensorIndex { i: i, j: j, k: k })
                                as usize;
                            let kernel_id = kernel.get_data_index(&TensorIndex {
                                i: a - stride * i,
                                j: b - stride * j,
                                k: c,
                            }) as usize;
                            convolution_result += kernel.data[kernel_id]
                                * incoming_gradients.data[incoming_gradient_id];
                        }
                    }
                }
                let outgoing_gradients_id =
                    outgoing_gradients.get_data_index(&TensorIndex { i: a, j: b, k: c }) as usize;

                outgoing_gradients.data[outgoing_gradients_id] = convolution_result;
            }
        }
    }
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
    check_convolution_dimensions(
        &image.get_shape(),
        &kernel_gradient.get_shape(),
        stride,
        &incoming_gradients.get_shape(),
        kernel_channel,
    )?;

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
    fn test_convolution_bias_relu() {
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
        let mut tensor_bias = Tensor::new(TensorShape::new_1d(1));
        tensor_bias.fill_with_value(-2.1);
        let mut tensor_result = Tensor::new(shape_result);
        let tensor_expected_result = Tensor {
            strides: TensorStride::new_from_shape(&shape_result),
            shape: shape_result,
            data: vec![3.0 - 2.1, 0.0, 0.0, 2.5 - 2.1],
        };
        assert!(convolution_bias_relu(
            &tensor_image,
            &tensor_kernel,
            &tensor_bias,
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
            .fill_with_vec(vec![1.0, 2.0, 0.0, 2.0])
            .expect("Could not fill incoming gradient tensor");
        let mut tensor_kernel_gradient = Tensor::new(shape_kernel);
        let mut tensor_expected_kernel_gradient = Tensor::new(shape_kernel);
        tensor_expected_kernel_gradient
            .fill_with_vec(vec![3.0, 2.0, 1.0, 2.0])
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
    fn test_convolution_backprop_outgoing_gradient() {
        let shape_input = TensorShape::new(3, 3, 1);
        let shape_kernel = TensorShape::new(2, 2, 1);
        let shape_result = TensorShape::new(2, 2, 2);
        let stride: u32 = 1;
        let mut tensor_image = Tensor::new(shape_input);
        tensor_image
            .fill_with_vec(vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0])
            .expect("Could not fill image tensor");
        let mut tensor_incoming_gradient = Tensor::new(shape_result);
        tensor_incoming_gradient
            .fill_with_vec(vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0])
            .expect("Could not fill incoming gradient tensor");

        // Kernels
        let mut kernel_1 = Tensor::new(shape_kernel);
        let mut kernel_2 = Tensor::new(shape_kernel);

        kernel_1
            .fill_with_vec(vec![1.0, 2.0, 3.0, 4.0])
            .expect("Could not fill kernel value");
        kernel_2
            .fill_with_vec(vec![8.0, 7.0, 6.0, 5.0])
            .expect("Could not fill kernel value");
        let kernels: Vec<Tensor> = vec![kernel_1, kernel_2];
        let mut tensor_outgoing_gradient = Tensor::new(shape_input);
        let mut tensor_expected_outgoing_gradient = Tensor::new(shape_input);
        tensor_expected_outgoing_gradient
            .fill_with_vec(vec![9.0, 11.0, 4.0, 9.0, 25.0, 19.0, 0.0, 12.0, 13.0])
            .expect("Could not fill incoming gradient tensor");
        assert!(convolution_backprop_outgoing_gradient(
            &tensor_incoming_gradient,
            &kernels,
            stride,
            &mut tensor_outgoing_gradient
        )
        .is_ok());
        assert_eq!(tensor_expected_outgoing_gradient, tensor_outgoing_gradient);
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
        )
        .is_ok());
        assert!(check_convolution_dimensions(
            &shape_image,
            &shape_kernel,
            stride + 1,
            &shape_result,
            result_channel
        )
        .is_err());
    }
}
