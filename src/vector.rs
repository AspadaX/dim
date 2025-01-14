use serde::{Serialize, Deserialize};
/// The type of data that is being vectorized. This enum represents the different
/// types of data that can be processed and vectorized in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// Image data type for processing image files and raw image data
    Image,
    /// Text data type for processing strings and text documents
    Text, 
    /// Audio data type for processing audio files and sound data
    Audio,
    /// Video data type for processing video files and motion picture data
    Video,
}

/// A vector that contains a vectorized data and the original data. This struct pairs
/// the original data with its vector representation and type information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector<T> {
    /// The vector representation of the data as floating point values
    vector: Vec<f32>,
    /// The original data being vectorized
    data: T,
    /// The type of the data being stored
    data_type: DataType,
}

/// Shared behaviors between `Vector` types. This trait defines the common operations
/// that can be performed on vectorized data regardless of the underlying data type.
pub trait VectorOperations<T> {
    /// Get the vector representation of the data
    /// 
    /// Returns a clone of the internal vector of f32 values
    fn get_vector(&self) -> Vec<f32>;

    /// Get a reference to the original data
    ///
    /// Returns an immutable reference to the stored data
    fn get_data(&self) -> &T;

    /// Get the type of data stored
    ///
    /// Returns the DataType enum indicating what kind of data is vectorized
    fn get_data_type(&self) -> DataType;

    /// Get dimensionality of the vector
    ///
    /// Returns the length of the vector representation
    fn get_dimensionality(&self) -> usize {
        self.get_vector().len()
    }

    /// Write a new vector to the vector field
    ///
    /// # Arguments
    /// * `vector` - The new vector to replace the existing one
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
    /// Initialize a new vector from image data
    ///
    /// # Arguments
    /// * `data` - The image data to be vectorized
    ///
    /// # Returns
    /// A new Vector instance containing the image data
    pub fn from_image(data: DynamicImage) -> Self {
        Self {
            vector: vec![],
            data,
            data_type: DataType::Image,
        }
    }
}

impl<String> Vector<String> {
    /// Initialize a new vector from text data
    ///
    /// # Arguments
    /// * `data` - The text data to be vectorized
    ///
    /// # Returns 
    /// A new Vector instance containing the text data
    pub fn from_text(data: String) -> Self {
        Self {
            vector: vec![],
            data,
            data_type: DataType::Text,
        }
    }
}
