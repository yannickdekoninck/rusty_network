use serde::{Deserialize, Serialize};
use std::cmp::Eq;
use std::collections::HashMap;

use crate::layer::Layer;
use crate::tensor::helper::TensorShape;
use crate::tensor::Tensor;

use super::SerializedLayer;

// using an enum for defining keys in serialized data
#[derive(Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum SoftMaxSerialKeys {
    Shape,
    Name,
}

pub struct SoftmaxLayer {
    shape: TensorShape,
    name: String,
}

impl SoftmaxLayer {
    pub fn new(shape: TensorShape, name: &String) -> Result<SoftmaxLayer, &'static str> {
        let mut sm = SoftmaxLayer::empty();
        sm.fill_from_state(shape, name)?;
        return Ok(sm);
    }

    pub fn empty() -> Self {
        let sm = SoftmaxLayer {
            shape: TensorShape {
                di: 1,
                dj: 1,
                dk: 1,
            },
            name: String::from("Empty softmax layer"),
        };

        return sm;
    }

    pub fn fill_from_state(
        self: &mut Self,
        shape: TensorShape,
        name: &String,
    ) -> Result<(), &'static str> {
        self.shape = shape;
        self.name = name.clone();
        return Ok(());
    }

    fn load_from_serialized_data(
        self: &mut Self,
        data: &HashMap<SoftMaxSerialKeys, Vec<u8>>,
    ) -> Result<(), &'static str> {
        // Check the correct keys are present
        if !data.contains_key(&SoftMaxSerialKeys::Name) {
            return Err("Data does not contain --name-- key");
        }
        if !data.contains_key(&SoftMaxSerialKeys::Shape) {
            return Err("Data does not contain --shape-- key");
        }

        let shape_data = data
            .get(&SoftMaxSerialKeys::Shape)
            .expect("Cannot access shape data");
        self.shape = bincode::deserialize(shape_data).expect("Cannot deserialize shape data");

        let name_data = data
            .get(&SoftMaxSerialKeys::Name)
            .expect("Cannot access name data");
        self.name = bincode::deserialize(name_data).expect("Cannot deserialize name data");

        return Ok(());
    }
}

impl Layer for SoftmaxLayer {
    fn forward(self: &mut Self, input: &Tensor, output: &mut Tensor) {
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
    fn get_serialized(self: &Self) -> SerializedLayer {
        // Create empty map
        let mut serial_data: HashMap<SoftMaxSerialKeys, Vec<u8>> = HashMap::new();
        // Serialize elements that need to be serialized
        let serial_shape = bincode::serialize(&self.shape).unwrap();
        let serial_name = bincode::serialize(&self.name).unwrap();
        // Add to hashmap
        serial_data.insert(SoftMaxSerialKeys::Shape, serial_shape);
        serial_data.insert(SoftMaxSerialKeys::Name, serial_name);
        // Return wrapped in softmax layer
        return SerializedLayer::SerialSoftMaxLayer(serial_data);
    }

    fn load_from_serialized(
        self: &mut Self,
        serial_data: &SerializedLayer,
    ) -> Result<(), &'static str> {
        // Unwrapping the serialized layer and checking it is the correct type
        match serial_data {
            SerializedLayer::SerialSoftMaxLayer(data) => {
                return self.load_from_serialized_data(data);
            }
            _ => {
                return Err("Layer type does not match soft max type");
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let sml = SoftmaxLayer::new(
            TensorShape {
                di: 5,
                dj: 4,
                dk: 3,
            },
            &String::from("Softmax layer"),
        )
        .unwrap();
        let serialized_sml = sml.get_serialized();

        let mut sml2 =
            SoftmaxLayer::new(TensorShape::new(1, 1, 1), &String::from("Nothing")).unwrap();
        assert!(sml2.load_from_serialized(&serialized_sml).is_ok());

        assert_eq!(sml2.name, String::from("Softmax layer"));
        assert_eq!(
            sml2.shape,
            TensorShape {
                di: 5,
                dj: 4,
                dk: 3,
            },
        );
    }
}
