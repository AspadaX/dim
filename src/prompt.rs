/// A prompt to be used for LLM-based vector generation
/// 
/// This struct represents an instruction prompt that will be sent to a Large Language Model
/// for generating vector representations based on the provided attribute description.
///
/// # Fields
/// * `instruction` - The formatted instruction string that will be sent to the LLM
pub struct Prompt {
    instruction: String
}

impl Prompt {
    /// Creates a new Prompt with the given attribute description
    ///
    /// # Arguments
    /// * `attribute_description` - A description of the attribute to evaluate
    ///
    /// # Returns
    /// A new Prompt instance configured with the formatted instruction
    pub fn new(attribute_description: String) -> Self {
        Self {
            instruction: format!(
                "output in json. Rate the text based on the guideline provided. Rate from 0.0 to 10.0. {{'offensiveness': your score}}\nGuideline: {}",
                attribute_description
            )
        }
    }

    /// Returns a clone of the instruction string
    ///
    /// # Returns
    /// The formatted instruction as a String
    pub fn get_instruction(&self) -> String {
        self.instruction.clone()
    }
}