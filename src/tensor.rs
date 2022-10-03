pub mod helper;

use helper::TensorIndex;
use helper::TensorShape;
use helper::TensorStride;

#[derive(Debug)]
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
        if Tensor::check_matrix_multiply_dimensions(tensor1, tensor2, result) {}
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
}

#[cfg(test)]
mod test {
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
}
