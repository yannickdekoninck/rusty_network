pub mod helper;

use helper::TensorIndex;
use helper::TensorShape;
use helper::TensorStride;

pub struct Tensor {
    strides: TensorStride,
    shape: TensorShape,
    data: Vec<f32>,
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

    fn get_data_index(self: &Self, index: &TensorIndex) -> u32 {
        return &self.strides * &index;
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
}
