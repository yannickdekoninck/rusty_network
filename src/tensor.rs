pub mod helper;

use helper::TensorIndex;
use helper::TensorShape;
use helper::TensorStride;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    strides: TensorStride,
    shape: TensorShape,
    data: Vec<f32>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        if self.data != other.data {
            return false;
        }
        return true;
    }

    fn ne(&self, other: &Self) -> bool {
        return !self.eq(other);
    }
}

impl Tensor {
    pub fn new(shape: TensorShape) -> Tensor {
        let total_size = shape.total_size();
        return Tensor {
            strides: TensorStride::new_from_shape(&shape),
            shape: shape,
            data: vec![0.0; total_size as usize],
        };
    }

    pub fn total_items(self: &Self) -> u32 {
        return self.shape.total_size();
    }

    pub fn set_item(self: &mut Self, index: &TensorIndex, value: f32) {
        if self.shape.fits_index(&index) {
            let data_index = self.get_data_index(index);
            self.data[data_index as usize] = value;
        }
    }

    pub fn get_item(self: &Self, index: &TensorIndex) -> f32 {
        let mut return_val: f32 = 0.0;
        if self.shape.fits_index(&index) {
            let data_index = self.get_data_index(index);
            return_val = self.data[data_index as usize];
        }
        return return_val;
    }

    pub fn get_data_index(self: &Self, index: &TensorIndex) -> u32 {
        return &self.strides * &index;
    }

    pub fn get_shape(self: &Self) -> TensorShape {
        return self.shape.clone();
    }

    pub fn fill_with_uniform(&mut self, min: f32, max: f32) {
        let uniform_distribution = rand::distributions::Uniform::new_inclusive(min, max);
        let mut rng = rand::prelude::thread_rng();
        for v in &mut self.data {
            *v = rng.sample(uniform_distribution);
        }
    }

    pub fn fill_with_gaussian(&mut self, mean: f32, std_dev: f32) {
        let uniform_distribution = rand_distr::Normal::new(mean, std_dev).unwrap();
        let mut rng = rand::prelude::thread_rng();
        for v in &mut self.data {
            *v = rng.sample(uniform_distribution);
        }
    }

    pub fn add(tensor1: &Tensor, tensor2: &Tensor, result: &mut Tensor) {
        if (tensor1.shape == tensor2.shape) && (tensor1.shape == result.shape) {
            // Create operants iterator
            let operants = tensor1.data.iter().zip(tensor2.data.iter());
            // Loop through items and calculate results
            for (rs, (op1, op2)) in result.data.iter_mut().zip(operants) {
                *rs = *op1 + *op2;
            }
        }
    }

    pub fn scale(tensor: &Tensor, scalar: f32, result: &mut Tensor) {
        if tensor.shape == result.shape {
            // Loop through items and calculate results
            for (rs, op) in result.data.iter_mut().zip(tensor.data.iter()) {
                *rs = *op * scalar;
            }
        }
    }
    pub fn matrix_multiply(tensor1: &Tensor, tensor2: &Tensor, result: &mut Tensor) {
        // Check if the dimensions allow for matrix multiplication
        let sr = result.shape.get();
        let s1 = tensor1.shape.get();
        if Tensor::check_matrix_multiply_dimensions(tensor1, tensor2, result) {
            // Loop over the outer dimension
            for k in 0..sr.2 {
                for j in 0..sr.1 {
                    for i in 0..sr.0 {
                        let mut running_result = 0.0;

                        // inner adder loop

                        for ii in 0..s1.1 {
                            let get_index1 =
                                tensor1.get_data_index(&TensorIndex { i: i, j: ii, k: k }) as usize;

                            let get_index2 =
                                tensor2.get_data_index(&TensorIndex { i: ii, j: j, k: k }) as usize;

                            running_result += tensor1.data[get_index1] * tensor2.data[get_index2];
                        }

                        let set_index =
                            result.get_data_index(&TensorIndex { i: i, j: j, k: k }) as usize;
                        result.data[set_index] = running_result;
                    }
                }
            }
        }
    }

    fn check_matrix_multiply_dimensions(
        tensor1: &Tensor,
        tensor2: &Tensor,
        result: &Tensor,
    ) -> bool {
        let st1 = tensor1.shape.get();
        let st2 = tensor2.shape.get();
        let sr = result.shape.get();
        // Outer dimensions must be the same
        if st1.2 != st2.2 || st1.2 != sr.2 {
            return false;
        }
        // Inner dimensions must match
        if st1.1 != st2.0 || st1.0 != sr.0 || st2.1 != sr.1 {
            return false;
        }
        return true;
    }

    pub fn matrix_multiply_add_relu(
        input: &Tensor,
        weights: &Tensor,
        bias: &Tensor,
        result: &mut Tensor,
    ) {
        // This is a helper function to do matrix multiply. bias add and relu in a single operation
        // This should save storing and loading of intermediate values
        // Everything should be nice and hot in the cache

        if Tensor::check_matrix_multiply_dimensions(weights, &input, &result) {
            // Check we can do the add
            if bias.shape == result.shape {
                let result_shape = result.shape.get();
                let weights_shape = weights.shape.get();
                // Loop over the outer dimension
                for k in 0..result_shape.2 {
                    for j in 0..result_shape.1 {
                        for i in 0..result_shape.0 {
                            let mut running_result = 0.0;

                            // inner matrix multiply loop

                            for ii in 0..weights_shape.1 {
                                let get_index1 =
                                    weights.get_data_index(&TensorIndex { i: i, j: ii, k: k })
                                        as usize;

                                let get_index2 =
                                    input.get_data_index(&TensorIndex { i: ii, j: j, k: k })
                                        as usize;

                                running_result += weights.data[get_index1] * input.data[get_index2];
                            }

                            // Add bias

                            let bias_index =
                                bias.get_data_index(&TensorIndex { i: i, j: j, k: k }) as usize;
                            running_result += bias.data[bias_index];

                            running_result = running_result.max(0.0);

                            let set_index =
                                result.get_data_index(&TensorIndex { i: i, j: j, k: k }) as usize;
                            result.data[set_index] = running_result;
                        }
                    }
                }
            }
        }
    }

    pub fn does_kernel_stride_fit_image(
        image_shape: &TensorShape,
        kernel_shape: &TensorShape,
        stride: u32,
    ) -> bool {
        if (image_shape.di - kernel_shape.di) % stride != 0 {
            return false;
        }
        if (image_shape.dj - kernel_shape.dj) % stride != 0 {
            return false;
        }

        return true;
    }

    pub fn get_convolution_dim_fit(
        image_shape: &TensorShape,
        kernel_shape: &TensorShape,
        stride: u32,
    ) -> (u32, u32) {
        let dim0_kernel_fits = (image_shape.di - kernel_shape.di) / stride + 1;
        let dim1_kernel_fits = (image_shape.dj - kernel_shape.dj) / stride + 1;
        return (dim0_kernel_fits, dim1_kernel_fits);
    }

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
    ) {
        if Tensor::check_convolution_dimensions(
            &image.get_shape(),
            &kernel.get_shape(),
            stride,
            &result.get_shape(),
            result_channel,
        ) {
            // All clear to convolute!

            let result_shape = result.shape.get();
            let kernel_shape = kernel.shape.get();

            // Main loop over image
            for j in 0..result_shape.1 {
                for i in 0..result_shape.0 {
                    let mut convolution_result: f32 = 0.0;
                    let image_start_i = i * stride;
                    let image_start_j = j * stride;

                    // Loop over kernel dimensions and multiply - add
                    for kk in 0..kernel_shape.2 {
                        for kj in 0..kernel_shape.1 {
                            for ki in 0..kernel_shape.0 {
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
        }
    }
}

#[cfg(test)]
mod test {
    use std::vec;

    use super::*;

    //
    #[test]
    fn test_create_tensor() {
        // Create 0 initialized tensor
        let t = Tensor::new(TensorShape::new(3, 4, 5));
        assert_eq!(t.total_items(), 60);
    }

    #[test]
    fn test_get_set_tensor() {
        let mut t = Tensor::new(TensorShape::new(3, 4, 5));
        let test_index = TensorIndex { i: 1, j: 3, k: 2 };
        let test_index2 = TensorIndex { i: 5, j: 3, k: 2 };
        // Check if tensor is initialized to 0
        assert_eq!(t.get_item(&test_index), 0.0);
        // Set tensor at index
        t.set_item(&test_index, 7.0);
        // Get tensor value at index and check it has changed
        assert_eq!(t.get_item(&test_index), 7.0);
        // Check we can't set something out of bounds
        t.set_item(&test_index2, 16.0);
        // Nothing should have changed to this value
        assert_eq!(t.get_item(&test_index2), 0.0);
    }

    #[test]
    fn test_uniform_fill() {
        let mut t = Tensor::new(TensorShape {
            di: 2,
            dj: 3,
            dk: 4,
        });
        t.fill_with_uniform(-1.0, 1.0);
        for d in t.data {
            assert!(d <= 1.0);
            assert!(d >= -1.0);
        }
    }

    #[test]
    fn test_tensor_equals() {
        let shape = TensorShape::new(2, 3, 1);
        let t1 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let t2 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };
        let t3 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        assert_eq!(t1, t3);
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_tensor_add() {
        let shape = TensorShape::new(2, 3, 1);
        let t1 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let t2 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };

        let mut result = Tensor::new(shape);

        let expected_result = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        };
        Tensor::add(&t1, &t2, &mut result);

        assert_eq!(expected_result, result);
    }

    #[test]
    fn test_tensor_scale() {
        let shape = TensorShape::new(2, 3, 1);
        let t1 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let scalar = 2.0;

        let mut result = Tensor::new(shape);

        let expected_result = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        };
        Tensor::scale(&t1, scalar, &mut result);

        assert_eq!(expected_result, result);
    }

    #[test]
    fn test_tensor_matrix_multiply() {
        let shape_1 = TensorShape::new(2, 3, 1);
        let shape_2 = TensorShape::new(3, 3, 1);
        let shape_res = TensorShape::new(2, 3, 1);
        let t1 = Tensor {
            shape: shape_1,
            strides: TensorStride::new_from_shape(&shape_1),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let t2 = Tensor {
            shape: shape_2,
            strides: TensorStride::new_from_shape(&shape_2),
            data: vec![1.0; 9],
        };

        let mut result = Tensor::new(shape_res);

        let expected_result = Tensor {
            shape: shape_res,
            strides: TensorStride::new_from_shape(&shape_res),
            data: vec![9.0, 12.0, 9.0, 12.0, 9.0, 12.0],
        };
        Tensor::matrix_multiply(&t1, &t2, &mut result);

        assert_eq!(expected_result, result);
    }
    #[test]
    fn test_matrix_multiply_shape_check() {
        let shape_1 = TensorShape::new(5, 6, 7);
        let shape_2 = TensorShape::new(6, 8, 7);
        let shape_res = TensorShape::new(5, 8, 7);
        let tensor_1 = Tensor::new(shape_1);
        let tensor_2 = Tensor::new(shape_2);
        let tensor_res = Tensor::new(shape_res);
        assert!(Tensor::check_matrix_multiply_dimensions(
            &tensor_1,
            &tensor_2,
            &tensor_res
        ));
        assert!(!Tensor::check_matrix_multiply_dimensions(
            &tensor_2,
            &tensor_1,
            &tensor_res
        ));
    }

    #[test]
    fn test_convolution_check_dimensions() {
        let shape_image = TensorShape::new(17, 25, 7);
        let shape_kernel = TensorShape::new(3, 3, 7);
        let shape_result = TensorShape::new(8, 12, 3);
        let stride: u32 = 2;
        let result_channel: u32 = 1;

        assert!(Tensor::check_convolution_dimensions(
            &shape_image,
            &shape_kernel,
            stride,
            &shape_result,
            result_channel
        ));
        assert!(!Tensor::check_convolution_dimensions(
            &shape_image,
            &shape_kernel,
            stride + 1,
            &shape_result,
            result_channel
        ));
    }

    #[test]
    fn test_convolution() {
        let shape_input = TensorShape::new(4, 4, 1);
        let shape_kernel = TensorShape::new(3, 3, 1);
        let shape_result = TensorShape::new(2, 2, 1);
        let stride: u32 = 1;
        let result_channel: u32 = 0;
        let tensor_image = Tensor {
            strides: TensorStride::new_from_shape(&shape_input),
            shape: shape_input,
            data: vec![
                1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5,
            ],
        };
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
        Tensor::convolution(
            &tensor_image,
            &tensor_kernel,
            stride,
            &mut tensor_result,
            result_channel,
        );
        assert_eq!(tensor_result, tensor_expected_result);
    }

    #[test]
    fn test_multiply_add_relu() {
        let shape_input = TensorShape::new(2, 1, 1);
        let shape_weights = TensorShape::new(3, 2, 1);
        let shape_bias = TensorShape::new(3, 1, 1);
        let shape_result = TensorShape::new(3, 1, 1);
        let tensor_input = Tensor {
            strides: TensorStride::new_from_shape(&shape_input),
            shape: shape_input,
            data: vec![1.0, 0.5],
        };
        let tensor_bias = Tensor {
            strides: TensorStride::new_from_shape(&shape_bias),
            shape: shape_bias,
            data: vec![1.0, 0.0, -1.0],
        };
        let tensor_weights = Tensor {
            strides: TensorStride::new_from_shape(&shape_weights),
            shape: shape_weights,
            data: vec![1.0, 0.0, 2.0, 2.0, -1.0, 0.0],
        };
        let mut tensor_result = Tensor::new(shape_result);
        let tensor_expected_result = Tensor {
            strides: TensorStride::new_from_shape(&shape_result),
            shape: shape_result,
            data: vec![3.0, 0.0, 1.0],
        };
        Tensor::matrix_multiply_add_relu(
            &tensor_input,
            &tensor_weights,
            &tensor_bias,
            &mut tensor_result,
        );
        assert_eq!(tensor_result, tensor_expected_result);
    }

    #[test]
    fn test_tensor_serialize_deserialize() {
        let tensor = Tensor::new(TensorShape {
            di: 5,
            dj: 5,
            dk: 5,
        });

        let serialized_tensor = bincode::serialize(&tensor).unwrap();

        let deserialized_tensor: Tensor = bincode::deserialize(&serialized_tensor[..]).unwrap();

        assert_eq!(deserialized_tensor, tensor);
    }
}
