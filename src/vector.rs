use image::{self, DynamicImage};

use crate::vectorizations::ImageVectorization;

/// currently supported datatypes
#[derive(Debug, Clone)]
pub enum SubjectType {
	Image,
	Text
}

#[derive(Debug, Clone)]
pub struct Vector<S> {
	// vectorized subject
	vector: Vec<f64>,
	// annotations to the `vector`
	annotations: Vec<String>,
	// prompts to be sent to the LLM for generating single elements
	prompts: Vec<String>,
	// how many vector elements that a single prompt will occupy with
	prompt_size: usize,
	// dimensionality of the `vector`
	dimensionality: usize,
	// anything that is going to be vectorized
	subject: S,
	// subject type
	subject_type: SubjectType
}

impl<S> Vector<S> {
	/// Creates a new Vector instance with the specified parameters
    ///
    /// # Arguments
    /// 
    /// * `dimensionality` - The size/length of the vector to be created
    /// * `annotations` - A collection of strings describing or labeling vector elements
    /// * `prompts` - Collection of prompts to be used for LLM-based vector generation
    /// * `prompt_size` - Number of vector elements that will be generated per prompt
    /// * `subject` - The source data to be vectorized
    ///
    /// # Returns
    ///
    /// Returns a new Vector instance with pre-allocated capacity but empty vector content
    ///
    /// # Example
    ///
    /// ```
    /// let vector = Vector::<f32, String>::new( // specify the vector element data type and the subject to be vectorized's datatype
    ///     100,                          // dimensionality
    ///     vec!["label1".to_string()],   // annotations
    ///     vec!["prompt1".to_string()],  // prompts
    ///     10,                           // prompt_size
    ///     "subject data"                // subject
    /// 	Image						  // type of the subject
    /// );
    /// ```
    pub fn new(
        dimensionality: usize, 
        annotations: Vec<String>,
        prompts: Vec<String>,
        prompt_size: usize,
        subject: S
    ) -> Self 
    where
    	S: Into<SubjectType> + Clone
    {
        // Create a new Vector instance with pre-allocated capacity
        let vector: Vec<f64> = Vec::with_capacity(dimensionality);
        Self {
            // Initialize empty vector with specified capacity
            vector: vector, 
            // Store vector element annotations
            annotations: annotations,
            // Store target dimensionality
            dimensionality: dimensionality,
            // Store LLM prompts
            prompts: prompts,
            // Store elements per prompt
            prompt_size: prompt_size,
            // Store source data
            subject: subject.clone(),
            // infer subject type from the subject
            subject_type: subject.into()
        }
    }
	
}

/// below are `Into` implementations for some of the common types

impl Into<SubjectType> for image::DynamicImage {
    fn into(self) -> SubjectType {
        SubjectType::Image
    }
}

impl Into<SubjectType> for String {
    fn into(self) -> SubjectType {
        SubjectType::Text
    }
}

impl Into<SubjectType> for &str {
    fn into(self) -> SubjectType {
        SubjectType::Text
    }
}

/// implement the vectoriazation trait of images 
impl<C> ImageVectorization<C> for Vector<DynamicImage> {
	
	fn update_vector(&mut self, vector: Vec<f64>) {
		self.vector = vector;
	}
	
	fn validate_vectorization_result(&self, vector: &Vec<f64>) -> bool {
		for element in vector {
			if element < &0.0 {
				return false;
			}
		}
		
		true
	}
	
}