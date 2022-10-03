use std::cmp;
use std::ops;
// Structure to define the shape of a tensor
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct TensorShape {
    di: u32,
    dj: u32,
    dk: u32,
}

impl TensorShape {
    pub fn new(di: u32, dj: u32, dk: u32) -> TensorShape {
        return TensorShape {
            di: cmp::max(di, 1),
            dj: cmp::max(dj, 1),
            dk: cmp::max(dk, 1),
        };
    }

    // Helper to create 1D tensor shape
    pub fn new_1d(di: u32) -> TensorShape {
        return TensorShape {
            di: di,
            dj: 1,
            dk: 1,
        };
    }

    // Helper to create 2D tensor shape
    pub fn new_2d(di: u32, dj: u32) -> TensorShape {
        return TensorShape {
            di: di,
            dj: dj,
            dk: 1,
        };
    }

    // Calculate the total size of the tensorshape
    pub fn total_size(self: &Self) -> u32 {
        return self.di * self.dj * self.dk;
    }

    pub fn fits_index(self: &Self, index: &TensorIndex) -> bool {
        // Check if an index fits in a certain shape
        if index.i >= self.di {
            return false;
        }
        if index.j >= self.dj {
            return false;
        }
        if index.k >= self.dk {
            return false;
        }
        return true;
    }

    pub fn get(self: &Self) -> (u32, u32, u32) {
        return (self.di, self.dj, self.dk);
    }
}

// Structure to index into a tensor
pub struct TensorIndex {
    pub i: u32,
    pub j: u32,
    pub k: u32,
}

impl TensorIndex {
    pub fn index_1d(i: u32) -> TensorIndex {
        return TensorIndex { i: i, j: 0, k: 0 };
    }

    pub fn index_2d(i: u32, j: u32) -> TensorIndex {
        return TensorIndex { i: i, j: j, k: 0 };
    }

    // A zero index
    pub fn zero() -> TensorIndex {
        return TensorIndex { i: 0, j: 0, k: 0 };
    }
}

#[derive(Debug, PartialEq)]
pub struct TensorStride {
    pub i: u32,
    pub j: u32,
    pub k: u32,
}

// Override multiplication operator
impl<'a> ops::Mul<&'a TensorIndex> for &'a TensorStride {
    type Output = u32;
    fn mul(self: &'a TensorStride, rhs: &'a TensorIndex) -> Self::Output {
        return self.i * rhs.i + self.j * rhs.j + self.k * rhs.k;
    }
}

impl TensorStride {
    pub fn new_from_shape(shape: &TensorShape) -> TensorStride {
        return TensorStride {
            i: 1,
            j: shape.di,
            k: shape.di * shape.dj,
        };
    }
}

// Unit tests

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensorshape_total_size() {
        // Testing the normal condition
        let ts = TensorShape::new(5, 4, 8);
        assert_eq!(ts.total_size(), 160);

        // Testing the ability to fix 0 dimensions
        let ts: TensorShape = TensorShape::new(0, 2, 0);
        assert_eq!(ts.total_size(), 2);
    }

    // Test is tensor stride object is properly constructed
    #[test]
    fn test_tensor_stride_from_shape() {
        let test_shape = TensorShape::new(5, 6, 10);
        let generated_stride = TensorStride::new_from_shape(&test_shape);
        assert_eq!(generated_stride, TensorStride { i: 1, j: 5, k: 30 });
    }

    #[test]
    fn test_index_fits_shape() {
        let test_shape = TensorShape::new(5, 6, 10);
        let test_index: TensorIndex = TensorIndex { i: 2, j: 4, k: 7 };
        let test_index2: TensorIndex = TensorIndex { i: 5, j: 4, k: 7 };

        assert_eq!(test_shape.fits_index(&test_index), true);
        assert_eq!(test_shape.fits_index(&test_index2), false);
    }

    #[test]
    fn test_index_stride_multiply() {
        let test_shape = TensorShape::new(5, 6, 10);
        let test_stride = TensorStride::new_from_shape(&test_shape);
        let test_index: TensorIndex = TensorIndex { i: 2, j: 4, k: 7 };
        let index = &test_stride * &test_index;
        assert_eq!(index, 232)
    }
}
