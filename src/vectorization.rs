use std::sync::Arc;

use anyhow::{Error, Result};
use async_openai::{config::Config, types::{ChatCompletionRequestMessageContentPartImageArgs, ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageDetail, ImageUrlArgs, ResponseFormat}, Client};
use base64::prelude::*;
use futures::future::join_all;
use image::DynamicImage;
use rand::Rng;
use serde_json::Value;

use crate::vector::{Vector, VectorOperations};

pub struct ModelParameters {
    model: String,
    temperature: f32,
    seed: i64,
}

impl ModelParameters {
    /// Creates a new instance of `ModelParameters`.
    ///
    /// # Arguments
    ///
    /// * `model` - A string representing the model name.
    /// * `temperature` - An optional floating-point value representing the temperature setting.
    /// * `seed` - An optional integer value representing the seed for random number generation.
    ///
    /// # Returns
    ///
    /// A new instance of `ModelParameters` with the specified model, temperature, and seed.
    ///
    /// If `temperature` is not provided, it defaults to 0.0.
    /// If `seed` is not provided, a random seed is generated.
    pub fn new(model: String, temperature: Option<f32>, seed: Option<i64>) -> Self {
        let temperature: f32 = temperature.unwrap_or(0.0);
        let seed: i64 = seed.unwrap_or_else(|| {
            let mut rng = rand::rng();
            rng.random()
        });

        Self {
            model,
            temperature,
            seed,
        }
    }

    pub fn get_model(&self) -> String {
        self.model.clone()
    }

    pub fn get_temperature(&self) -> f32 {
        self.temperature
    }

    pub fn get_seed(&self) -> i64 {
        self.seed
    }
}

/// Converts a DynamicImage to a base64-encoded string
fn dynamic_image_to_base64(image: &DynamicImage) -> Result<String, Error> {
    let mut raw_image_bytes: Vec<u8> = Vec::new();
    image.write_to(
        &mut std::io::Cursor::new(&mut raw_image_bytes),
        image::ImageFormat::Png,
    )?;
    let base64_image: String = BASE64_STANDARD.encode(raw_image_bytes);

    Ok(base64_image)
}

/// Recursively extracts leaf values from a JSON response retrieved from the LLM.
/// 
/// Takes a JSON Value and returns a Vec of all leaf values found in the structure.
fn extract_leaf_values_recursively(value: &Value) -> Vec<Value> {
    match value {
        Value::Object(map) => map
            .values()
            .flat_map(|v| extract_leaf_values_recursively(v))
            .collect(),
        Value::Array(arr) => arr
            .iter()
            .flat_map(|v| extract_leaf_values_recursively(v))
            .collect(),
        _ => vec![value.clone()],
    }
}

/// Validates that all elements in the vectorization result are non-negative.
/// 
/// Takes a vector slice and validates that it:
/// - Is not empty
/// - Contains exactly one element 
/// - All elements are non-negative (>= 0)
///
/// # Arguments
/// * `vector` - Vector slice to validate
///
/// # Returns 
/// * `bool` - True if vector meets all validation criteria, false otherwise
fn validate_vectorization_result(vector: &Vec<f32>) -> Result<(), Error> {
    // Return error if vector is empty
    if vector.is_empty() {
        return Err(Error::msg("Validation error: vector is empty"));
    // Check if vector has more than one element
    } else if vector.len() > 1 {
        return Err(Error::msg("Validation error: vector has more than one element"));
    }

    // Check if any elements are negative
    for element in vector {
        if *element < 0.0 {
            return Err(Error::msg("Validation error: vector contains negative elements"));
        }
    }

    // All validation checks passed
    Ok(())
}

/// Processes a single image with one prompt to generate a vector representation.
/// 
/// Continues retrying until valid results are obtained.
async fn vectorize_image_single_prompt<C>(
    client: &Client<C>,
    image: &DynamicImage,
    prompt: String,
    model_parameters: &ModelParameters,
) -> Result<Vec<f32>, Error>
where
    C: Config + Send + Sync + 'static,
{
    let base64_image = dynamic_image_to_base64(&image)?;
    let image_url = format!("data:image/jpeg;base64,{}", base64_image);

    loop {
        let request = match CreateChatCompletionRequestArgs::default()
            .temperature(model_parameters.get_temperature())
            .seed(model_parameters.get_seed())
            .model(model_parameters.get_model())
            .response_format(ResponseFormat::JsonObject)
            .messages(vec![ChatCompletionRequestUserMessageArgs::default()
                .content(vec![
                    ChatCompletionRequestMessageContentPartTextArgs::default()
                        .text(&prompt)
                        .build()
                        .map_err(|e| Error::msg(e.to_string()))?
                        .into(),
                    ChatCompletionRequestMessageContentPartImageArgs::default()
                        .image_url(
                            ImageUrlArgs::default()
                                .url(&image_url)
                                .detail(ImageDetail::High)
                                .build()
                                .map_err(|e| Error::msg(e.to_string()))?,
                        )
                        .build()
                        .map_err(|e| Error::msg(e.to_string()))?
                        .into(),
                ])
                .build()
                .map_err(|e| Error::msg(e.to_string()))?
                .into()])
            .build()
        {
            Ok(req) => req,
            Err(e) => {
                println!("Failed to build request: {}", e);
                continue;
            }
        };

        let response = match client.chat().create(request).await {
            Ok(res) => res,
            Err(e) => {
                println!("API request error: {}", e);
                continue;
            }
        };

        let content = match response.choices.get(0).and_then(|c| c.message.content.as_ref()) {
            Some(c) => c,
            None => {
                println!("Empty content in response");
                continue;
            }
        };

        let parsed_json = match serde_json::from_str::<Value>(content) {
            Ok(v) => v,
            Err(e) => {
                println!("JSON parsing failed: {}", e);
                continue;
            }
        };

        let leaf_values = extract_leaf_values_recursively(&parsed_json);
        let result: Vec<f32> = leaf_values
            .into_iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if let Err(e) = validate_vectorization_result(&result) {
            println!("Validation failed: {}, retrying...", e);
            println!("Prompt: {}", prompt);
            println!("Result: {}", &parsed_json);
            println!("Output: {:?}", result);
        } else {
            return Ok(result);
        }
    }
}

/// Concurrently vectorizes an image with multiple prompts.
/// 
/// # Arguments
/// * `model` - The name/identifier of the LLM model to use
/// * `prompts` - A vector of prompts to process concurrently
/// * `vector` - A mutable reference to the Vector struct containing the image
/// * `client` - The OpenAI API client
/// 
/// # Returns
/// * `Result<(), Error>` - Ok(()) on success, Error on failure
/// 
/// Each prompt's dimensionality is specified by how many digits that it 
/// requires the LLM to return. The final dimensionality of the vector is 
/// calculated by `number of prompts * digits specified by each prompt`.
pub async fn vectorize_image_concurrently<C>(
    prompts: Vec<String>,
    vector: &mut Vector<DynamicImage>, 
    client: Client<C>,
    model_parameters: ModelParameters,
) -> Result<(), Error>
where
    C: Config + Send + Sync + 'static,
{
    // get data from the struct
    let image: DynamicImage = vector.get_data().clone();

    let shared_client: Arc<Client<C>> = Arc::new(client);
    let shared_image: Arc<DynamicImage> = Arc::new(image);
    let shared_model: Arc<ModelParameters> = Arc::new(model_parameters);

    // collect all tasks for concurrent execution
    let mut tasks = Vec::new();
    for (index, prompt) in prompts.into_iter().enumerate() {
        let shared_client: Arc<Client<C>> = shared_client.clone();
        let shared_image: Arc<DynamicImage> = shared_image.clone();
        let shared_model: Arc<ModelParameters> = shared_model.clone();

        let task = tokio::spawn(async move {
            let subvector: Vec<f32> = vectorize_image_single_prompt(
                shared_client.as_ref(), 
                shared_image.as_ref(), 
                prompt,
                shared_model.as_ref(),
            )
                .await?;
            println!("thread {index} finished vectorization.");

            Ok::<_, Error>(subvector)
        });

        tasks.push(task);
    }

    let results = join_all(tasks).await;

    // Collect and join the subvectors sequentially
    let final_vector: Vec<f32> = results
        .into_iter()
        .filter_map(|result| result.ok())
        .filter_map(|result| result.ok())
        .flat_map(|subvec| subvec.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        .collect();

    vector.overwrite_vector(final_vector);

    Ok(())
}

/// Processes a single text string with one prompt to generate a vector representation.
/// 
/// Continues retrying until valid results are obtained.
async fn vectorize_string_single_prompt<C>(
    client: &Client<C>,
    text: &str,
    prompt: String,
    model_parameters: &ModelParameters
) -> Result<Vec<f32>, Error>
where
    C: Config + Send + Sync + 'static,
{
    loop {
        let request: async_openai::types::CreateChatCompletionRequest = match CreateChatCompletionRequestArgs::default()
            .temperature(model_parameters.get_temperature())
            .seed(model_parameters.get_seed())
            .model(model_parameters.get_model())
            .response_format(ResponseFormat::JsonObject)
            .messages(vec![ChatCompletionRequestUserMessageArgs::default()
                .content(format!("{}\n\nText to analyze: {}", prompt, text))
                .build()
                .map_err(|e| Error::msg(e.to_string()))?
                .into()])
            .build()
        {
            Ok(req) => req,
            Err(e) => {
                println!("Failed to build request: {}", e);
                continue;
            }
        };

        let response = match client.chat().create(request).await {
            Ok(res) => res,
            Err(e) => {
                println!("API request error: {}", e);
                continue;
            }
        };

        let content = match response.choices.get(0).and_then(|c| c.message.content.as_ref()) {
            Some(c) => c,
            None => {
                println!("Empty content in response");
                continue;
            }
        };

        let parsed_json = match serde_json::from_str::<Value>(content) {
            Ok(v) => v,
            Err(e) => {
                println!("JSON parsing failed: {}", e);
                continue;
            }
        };

        let leaf_values = extract_leaf_values_recursively(&parsed_json);
        let result: Vec<f32> = leaf_values
            .into_iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if let Err(e) = validate_vectorization_result(&result) {
            println!("Validation failed: {}, retrying...", e);
            println!("Prompt: {}", prompt);
            println!("Text: {}", text);
            println!("Output: {:?}", result);
        } else {
            return Ok(result);
        }
    }
}

/// Concurrently vectorizes a text string with multiple prompts.
/// 
/// # Arguments
/// * `model` - The name/identifier of the LLM model to use
/// * `prompts` - A vector of prompts to process concurrently
/// * `vector` - A mutable reference to the Vector struct containing the text
/// * `client` - The OpenAI API client
/// 
/// # Returns
/// * `Result<(), Error>` - Ok(()) on success, Error on failure
pub async fn vectorize_string_concurrently<C>(
    prompts: Vec<String>,
    vector: &mut Vector<String>,
    client: Client<C>,
    model_parameters: ModelParameters,
) -> Result<(), Error>
where
    C: Config + Send + Sync + 'static,
{
    // get data from the struct
    let text: String = vector.get_data().clone();

    let shared_client: Arc<Client<C>> = Arc::new(client);
    let shared_text: Arc<String> = Arc::new(text);
    let shared_model: Arc<ModelParameters> = Arc::new(model_parameters);

    // collect all tasks for concurrent execution
    let mut tasks = Vec::new();
    for (index, prompt) in prompts.into_iter().enumerate() {
        let shared_client: Arc<Client<C>> = shared_client.clone();
        let shared_text: Arc<String> = shared_text.clone();
        let shared_model: Arc<ModelParameters> = shared_model.clone();

        let task = tokio::spawn(async move {
            let subvector = vectorize_string_single_prompt(
                shared_client.as_ref(),
                shared_text.as_ref(),
                prompt,
                shared_model.as_ref(),
            )
                .await?;
            println!("thread {index} finished vectorization.");

            Ok::<_, Error>(subvector)
        });

        tasks.push(task);
    }

    let results = join_all(tasks).await;

    // Collect and join the subvectors sequentially
    let final_vector: Vec<f32> = results
        .into_iter()
        .filter_map(|result| result.ok())
        .filter_map(|result| result.ok())
        .flat_map(|subvec| subvec.iter().map(|&x| x as f32).collect::<Vec<f32>>())
        .collect();

    vector.overwrite_vector(final_vector);

    Ok(())
}
