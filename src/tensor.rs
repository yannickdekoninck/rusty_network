pub mod helper;

use helper::TensorIndex;
use helper::TensorShape;
use helper::TensorStride;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

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

    pub fn empty() -> Tensor {
        let tensor_shape = TensorShape {
            di: 1,
            dj: 1,
            dk: 1,
        };
        return Tensor::new(tensor_shape);
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

    pub fn fill_with_value(&mut self, value: f32) {
        for v in &mut self.data {
            *v = value;
        }
    }

    pub fn fill_with_vec(&mut self, data: Vec<f32>) -> Result<(), &'static str> {
        if data.len() != (self.total_items() as usize) {
            return Err("Vec data did not contain the correct number of items");
        }
        self.data = data;
        return Ok(());
    }

    pub fn save_to_file(self: &Self, filename: &String) -> Result<(), Box<dyn Error>> {
        let serialized_tensor = bincode::serialize(self)?;
        fs::write(filename, serialized_tensor)?;
        return Ok(());
    }

    pub fn from_file(filename: &String) -> Result<Tensor, Box<dyn Error>> {
        let file_content = fs::read(filename)?;
        let tensor: Tensor = bincode::deserialize(&file_content)?;
        return Ok(tensor);
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
    pub fn add_to_self(tensor1: &mut Tensor, tensor2: &Tensor) {
        if tensor1.shape == tensor2.shape {
            // Create operants iterator
            let operants = tensor1.data.iter_mut().zip(tensor2.data.iter());
            // Loop through items and calculate results
            for (op1, op2) in operants {
                *op1 = *op1 + *op2;
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
    pub fn multiply_elementwise(tensor1: &Tensor, tensor2: &Tensor, result: &mut Tensor) {
        if (tensor1.shape == tensor2.shape) && (tensor1.shape == result.shape) {
            // Create operants iterator
            let operants = tensor1.data.iter().zip(tensor2.data.iter());
            // Loop through items and calculate results
            for (rs, (op1, op2)) in result.data.iter_mut().zip(operants) {
                *rs = *op1 * *op2;
            }
        }
    }

    pub fn relu_self_and_store_mask(
        input: &mut Tensor,
        mask: &mut Tensor,
    ) -> Result<(), &'static str> {
        if input.get_shape() != mask.get_shape() {
            return Err("relu_self_and_store_mask: Input and mask must have the same shape");
        }
        for (in_data, mask_data) in input.data.iter_mut().zip(mask.data.iter_mut()) {
            if *in_data > 0.0 {
                *mask_data = 1.0;
            } else {
                {
                    *in_data = 0.0;
                    *mask_data = 0.0;
                }
            }
        }
        return Ok(());
    }

    pub fn transpose_ij(input: &Tensor, output: &mut Tensor) -> Result<(), &'static str> {
        // Check dimensions
        let input_shape = input.get_shape();
        let output_shape = output.get_shape();
        if input_shape.dk != output_shape.dk {
            return Err("Third dimension must match for transpose ij");
        }
        if input_shape.di != output_shape.dj {
            return Err("i and j of input and output must match");
        }
        if input_shape.dj != output_shape.di {
            return Err("j and i of input and output must match");
        }
        // Transpose loop

        for k in 0..output_shape.dk {
            for j in 0..output_shape.dj {
                for i in 0..output_shape.di {
                    let output_id =
                        output.get_data_index(&TensorIndex { i: i, j: j, k: k }) as usize;
                    let input_id = input.get_data_index(&TensorIndex { i: j, j: i, k: k }) as usize;
                    output.data[output_id] = input.data[input_id];
                }
            }
        }

        return Ok(());
    }

    pub fn matrix_multiply_transpose_second(
        tensor1: &Tensor,
        tensor2: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), &'static str> {
        // This is simular to matrix multiply but first transposes transposes tensor2 before doing the multiplication
        // Check if the dimensions allow for matrix multiplication
        let sr = result.shape.get();
        let s1 = tensor1.shape.get();
        let shape_1 = tensor1.get_shape();
        let shape_2 = tensor2.get_shape();
        let shape_r = result.get_shape();
        // Check tensor shape with transposed tensor
        if !Tensor::check_matrix_multiply_dimensions(shape_1, shape_2.transpose_ij(), shape_r) {
            return Err("Shapes do not match for matrix multiply transpose");
        }
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
                            tensor2.get_data_index(&TensorIndex { i: j, j: ii, k: k }) as usize;

                        running_result += tensor1.data[get_index1] * tensor2.data[get_index2];
                    }

                    let set_index =
                        result.get_data_index(&TensorIndex { i: i, j: j, k: k }) as usize;
                    result.data[set_index] = running_result;
                }
            }
        }
        return Ok(());
    }
    pub fn matrix_multiply(tensor1: &Tensor, tensor2: &Tensor, result: &mut Tensor) {
        // Check if the dimensions allow for matrix multiplication
        let sr = result.shape.get();
        let s1 = tensor1.shape.get();
        if Tensor::check_matrix_multiply_dimensions(
            tensor1.get_shape(),
            tensor2.get_shape(),
            result.get_shape(),
        ) {
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
        st1: TensorShape,
        st2: TensorShape,
        sr: TensorShape,
    ) -> bool {
        // Outer dimensions must be the same
        if st1.dk != st2.dk || st1.dk != sr.dk {
            return false;
        }
        // Inner dimensions must match
        if st1.dj != st2.di || st1.di != sr.di || st2.dj != sr.dj {
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

        if Tensor::check_matrix_multiply_dimensions(
            weights.get_shape(),
            input.get_shape(),
            result.get_shape(),
        ) {
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
                            let set_index =
                                result.get_data_index(&TensorIndex { i: i, j: j, k: k }) as usize;

                            running_result += bias.data[bias_index];

                            running_result = running_result.max(0.0);

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

    pub fn check_max_pooling_dimensions(
        image_shape: &TensorShape,
        mask_shape: &TensorShape,
        stride: u32,
        result_shape: &TensorShape,
    ) -> bool {
        // We check and immediately exit when something is not right
        // No need to check anything else

        // image and result depth must match
        if result_shape.dk != image_shape.dk {
            return false;
        }

        // Check if the mask fitst in the image given the stride
        // We assume that the image is properly padded
        if (image_shape.di - mask_shape.di) % stride != 0 {
            return false;
        }
        if (image_shape.dj - mask_shape.dj) % stride != 0 {
            return false;
        }

        // Check if the output dimensions are correct
        let dim0_mask_fits = (image_shape.di - mask_shape.di) / stride + 1;
        let dim1_mask_fits = (image_shape.dj - mask_shape.dj) / stride + 1;
        if result_shape.di != dim0_mask_fits {
            return false;
        }
        if result_shape.dj != dim1_mask_fits {
            return false;
        }
        return true;
    }

    pub fn max_pool(image: &Tensor, mask_shape: &TensorShape, stride: u32, result: &mut Tensor) {
        if Tensor::check_max_pooling_dimensions(
            &image.get_shape(),
            mask_shape,
            stride,
            &result.get_shape(),
        ) {
            let result_shape = result.get_shape();
            // Max pooling loop
            for k in 0..result_shape.dk {
                for j in 0..result_shape.dj {
                    for i in 0..result_shape.di {
                        let mut running_max = f32::NEG_INFINITY;

                        for jj in 0..mask_shape.dj {
                            for ii in 0..mask_shape.di {
                                let current_item = image.get_item(&TensorIndex {
                                    i: i * stride + ii,
                                    j: j * stride + jj,
                                    k,
                                });
                                if current_item > running_max {
                                    running_max = current_item
                                }
                            }
                        }
                        result.set_item(&TensorIndex { i: i, j: j, k: k }, running_max);
                    }
                }
            }
        }
    }

    pub fn softmax(input: &Tensor, output: &mut Tensor) {
        if input.get_shape() == output.get_shape() {
            let mut sum = 0.0;
            for i in 0..input.data.len() {
                let exp = input.data[i].exp();
                output.data[i] = exp;
                sum += exp;
            }
            for f in output.data.iter_mut() {
                *f = *f / sum;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::vec;

    use super::*;
    use tempdir::TempDir;

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
    fn test_value_fill() {
        let mut t = Tensor::new(TensorShape {
            di: 2,
            dj: 3,
            dk: 4,
        });
        t.fill_with_value(7.0);
        for d in t.data {
            assert!(d == 7.0);
        }
    }

    #[test]
    fn test_vec_fill() {
        let mut t = Tensor::new(TensorShape {
            di: 2,
            dj: 2,
            dk: 2,
        });
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(t.fill_with_vec(data).is_ok());
        assert_eq!(t.get_item(&TensorIndex { i: 0, j: 1, k: 1 }), 7.0);
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
    fn test_tensor_add_to_self() {
        let shape = TensorShape::new(2, 3, 1);
        let mut t1 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let t2 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };

        let expected_result = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        };
        Tensor::add_to_self(&mut t1, &t2);

        assert_eq!(expected_result, t1);
    }
    #[test]
    fn test_tensor_relu_self_and_store_mask() {
        let shape = TensorShape::new(7, 1, 1);
        let mut input = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, -3.0, -4.0, 5.0, -6.0, 7.0],
        };
        let mut mask = Tensor::new(shape);

        let expected_mask = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        };
        assert!(Tensor::relu_self_and_store_mask(&mut input, &mut mask).is_ok());

        assert_eq!(expected_mask, mask);
    }
    #[test]
    fn test_tensor_multiply_elementwise() {
        let mut t1 = Tensor::new(TensorShape::new_2d(2, 3));
        let mut t2 = Tensor::new(TensorShape::new_2d(2, 3));
        let mut result = Tensor::new(TensorShape::new_2d(2, 3));
        let mut expected_result = Tensor::new(TensorShape::new_2d(2, 3));

        t1.fill_with_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        t2.fill_with_vec(vec![2.0, 1.0, 3.0, 5.0, 2.0, 1.0])
            .unwrap();
        expected_result
            .fill_with_vec(vec![2.0, 2.0, 9.0, 20.0, 10.0, 6.0])
            .unwrap();

        Tensor::multiply_elementwise(&t1, &t2, &mut result);

        assert_eq!(expected_result, result);
    }
    #[test]
    fn test_tensor_transpose_ij() {
        let input_shape = TensorShape::new(2, 3, 2);
        let output_shape = TensorShape::new(3, 2, 2);
        let input = Tensor {
            shape: input_shape,
            strides: TensorStride::new_from_shape(&input_shape),
            data: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        };
        let expected_output = Tensor {
            shape: output_shape,
            strides: TensorStride::new_from_shape(&output_shape),
            data: vec![
                1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 8.0, 10.0, 12.0,
            ],
        };

        let mut output = Tensor::new(output_shape);

        assert!(Tensor::transpose_ij(&input, &mut output).is_ok());

        assert_eq!(expected_output, output);
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
    fn test_softmax() {
        let shape = TensorShape::new(1, 3, 1);
        let t1 = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![1.0, 2.0, 3.0],
        };

        let mut result = Tensor::new(shape);
        let f1: f32 = (1.0 as f32).exp();
        let f2: f32 = (2.0 as f32).exp();
        let f3: f32 = (3.0 as f32).exp();
        let sum = f1 + f2 + f3;
        let expected_result = Tensor {
            shape: shape,
            strides: TensorStride::new_from_shape(&shape),
            data: vec![f1 / sum, f2 / sum, f3 / sum],
        };
        Tensor::softmax(&t1, &mut result);

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
    fn test_tensor_matrix_multiply_transpose_second() {
        let shape_1 = TensorShape::new(3, 1, 1);
        let shape_2 = TensorShape::new(3, 1, 1);
        let shape_res = TensorShape::new(3, 3, 1);
        let t1 = Tensor {
            shape: shape_1,
            strides: TensorStride::new_from_shape(&shape_1),
            data: vec![1.0, 2.0, 3.0],
        };
        let t2 = Tensor {
            shape: shape_2,
            strides: TensorStride::new_from_shape(&shape_2),
            data: vec![3.0, 4.0, 5.0],
        };

        let mut result = Tensor::new(shape_res);

        let expected_result = Tensor {
            shape: shape_res,
            strides: TensorStride::new_from_shape(&shape_res),
            data: vec![3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0],
        };
        assert!(Tensor::matrix_multiply_transpose_second(&t1, &t2, &mut result).is_ok());

        assert_eq!(expected_result, result);
    }
    #[test]
    fn test_matrix_multiply_shape_check() {
        let shape_1 = TensorShape::new(5, 6, 7);
        let shape_2 = TensorShape::new(6, 8, 7);
        let shape_res = TensorShape::new(5, 8, 7);
        assert!(Tensor::check_matrix_multiply_dimensions(
            shape_1.clone(),
            shape_2.clone(),
            shape_res.clone()
        ));
        assert!(!Tensor::check_matrix_multiply_dimensions(
            shape_2.clone(),
            shape_1.clone(),
            shape_res.clone()
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
    fn test_max_pooling() {
        let shape_input = TensorShape::new(5, 5, 3);
        let shape_mask = TensorShape::new(3, 3, 1);
        let shape_result = TensorShape::new(2, 2, 3);
        let stride: u32 = 2;
        let mut tensor_image = Tensor::new(shape_input);
        tensor_image.fill_with_value(1.0);
        tensor_image.set_item(&TensorIndex { i: 0, j: 0, k: 0 }, 2.0);
        tensor_image.set_item(&TensorIndex { i: 2, j: 2, k: 1 }, 3.0);
        tensor_image.set_item(&TensorIndex { i: 4, j: 4, k: 2 }, 4.0);
        let mut tensor_result = Tensor::new(shape_result);
        let tensor_expected_result = Tensor {
            strides: TensorStride::new_from_shape(&shape_result),
            shape: shape_result,
            data: vec![2.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 4.0],
        };
        Tensor::max_pool(&tensor_image, &shape_mask, stride, &mut tensor_result);

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

    #[test]
    fn test_write_read_tensor() {
        let dir = TempDir::new("test_tensor_write_read").unwrap();
        let file_path = dir.path().join("tensor.tsr");

        let mut tensor = Tensor::new(TensorShape {
            di: 3,
            dj: 4,
            dk: 5,
        });
        tensor.fill_with_gaussian(0.0, 1.0);
        tensor
            .save_to_file(&String::from(file_path.to_str().unwrap()))
            .unwrap();

        let other_tensor = Tensor::from_file(&String::from(file_path.to_str().unwrap())).unwrap();

        assert_eq!(tensor, other_tensor);
        dir.close().unwrap();
    }
}
