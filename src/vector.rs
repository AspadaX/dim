/// The type of data that is being vectorized
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Image,
    Text,
    Audio,
    Video,
}

/// A vector that contains a vectorized data and the original data
#[derive(Debug, Clone)]
pub struct Vector<T> {
    vector: Vec<f32>,
    data: T,
    data_type: DataType,
}

/// Shared behaviors between `Vector` types
pub trait VectorOperations<T> {
    /// get the vector
    fn get_vector(&self) -> Vec<f32>;

    /// get the data
    fn get_data(&self) -> &T;

    /// get the data type
    fn get_data_type(&self) -> DataType;

    /// get dimensionality of the vector
    fn get_dimensionality(&self) -> usize {
        self.get_vector().len()
    }
    
    /// write a new vector to the vector field
    fn overwrite_vector(&mut self, vector: Vec<f32>);
}

impl<T> VectorOperations<T> for Vector<T> {
    fn get_vector(&self) -> Vec<f32> {
        self.vector.clone()
    }

    fn get_data_type(&self) -> DataType {
        self.data_type
    }

    fn get_data(&self) -> &T {
        &self.data
    }
    
    fn overwrite_vector(&mut self, vector: Vec<f32>) {
        self.vector = vector;
    }
}

impl<DynamicImage> Vector<DynamicImage> {
    /// initialize a new vector
    pub fn from_image(data: DynamicImage) -> Self {
        Self {
            vector: vec![],
            data,
            data_type: DataType::Image,
        }
    }
}

impl<String> Vector<String> {
    /// initialize a new vector
    pub fn from_text(data: String) -> Self {
        Self {
            vector: vec![],
            data,
            data_type: DataType::Text,
        }
    }
}
