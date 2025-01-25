use std::sync::Arc;

use anyhow::{Error, Result};
use async_openai::{config::Config, types::{ChatCompletionRequestMessageContentPartImageArgs, ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ImageDetail, ImageUrlArgs, ResponseFormat}, Client};
use base64::prelude::*;
use futures::future::join_all;
use image::DynamicImage;
use serde_json::Value;

use crate::vector::{Vector, VectorOperations};

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
fn validate_vectorization_result(vector: &Vec<f32>) -> bool {
    // Check if vector is empty
    if vector.len() == 0 {
        return false;
    // Check if vector has more than one element
    } else if vector.len() > 1 {
        return false;
    }

    // Check if any elements are negative
    for element in vector {
        if element < &0.0 {
            return false;
        }
    }

    // All validation checks passed
    true
}

/// Processes a single image with one prompt to generate a vector representation.
/// 
/// Continues retrying until valid results are obtained.
async fn vectorize_image_single_prompt<C>(
    client: &Client<C>,
    model: &str,
    image: &DynamicImage,
    prompt: String,
) -> Result<Vec<f32>, Error>
where
    C: Config + Send + Sync + 'static,
{
    let base64_image: String = dynamic_image_to_base64(&image)?;
    let image: String = format!("data:image/jpeg;base64,{base64_image}");

    loop {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .response_format(ResponseFormat::JsonObject)
            .messages(vec![ChatCompletionRequestUserMessageArgs::default()
                .content(vec![
                    ChatCompletionRequestMessageContentPartTextArgs::default()
                        .text(&prompt)
                        .build()?
                        .into(),
                    ChatCompletionRequestMessageContentPartImageArgs::default()
                        .image_url(
                            ImageUrlArgs::default()
                                .url(&image)
                                .detail(ImageDetail::High)
                                .build()?,
                        )
                        .build()?
                        .into(),
                ])
                .build()?
                .into()])
            .build()?;

        // Send the request and await the response
        let response = match client
            .chat()
            .create(request)
            .await {
                Ok(result) => result,
                Err(error) => {
                    println!("Error: {:?}", error);
                    continue;
                }
            };

        // Get the content from the first choice
        let content = &response.choices[0].message.content.clone().unwrap();

        // Parse the content as JSON into ResponseData
        // data validations, if not validated, retry until succeed.
        let result: Vec<f32> = extract_leaf_values_recursively(
            &serde_json::from_str(content)?
        )
            .into_iter()
            .map(|score| score.as_f64().unwrap_or(0.0) as f32)
            .collect();

        // break the loop if the results are validated
        if validate_vectorization_result(&result) {
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
    model: &str,
    prompts: Vec<String>,
    vector: &mut Vector<DynamicImage>, 
    client: Client<C>
) -> Result<(), Error>
where
    C: Config + Send + Sync + 'static,
{
    // get data from the struct
    let image: DynamicImage = vector.get_data().clone();

    let shared_client: Arc<Client<C>> = Arc::new(client);
    let shared_image: Arc<DynamicImage> = Arc::new(image);
    let shared_model: Arc<String> = Arc::new(model.to_string());

    // collect all tasks for concurrent execution
    let mut tasks = Vec::new();
    for (index, prompt) in prompts.into_iter().enumerate() {
        let shared_client: Arc<Client<C>> = shared_client.clone();
        let shared_image: Arc<DynamicImage> = shared_image.clone();
        let shared_model: Arc<String> = shared_model.clone();

        let task = tokio::spawn(async move {
            let subvector: Vec<f32> = vectorize_image_single_prompt(
                shared_client.as_ref(), 
                shared_model.as_ref(),
                shared_image.as_ref(), 
                prompt
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
    model: &str,
    text: &str,
    prompt: String,
) -> Result<Vec<f32>, Error>
where
    C: Config + Send + Sync + 'static,
{
    loop {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .response_format(ResponseFormat::JsonObject)
            .messages(
                vec![
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(
                            format!("{}\n\nText to analyze: {}", prompt, text)
                        )
                        .build()?
                        .into()
                ]
            )
            .build()?;

        // Send the request and await the response
        let response = client.chat().create(request).await?;

        // Get the content from the first choice
        let content = &response.choices[0].message.content.clone().unwrap();

        // Parse the content as JSON into ResponseData
        let result: Vec<f32> = extract_leaf_values_recursively(
            &serde_json::from_str(content)?
        )
            .into_iter()
            .map(|score| score.as_f64().unwrap_or(0.0) as f32)
            .collect();

        // break the loop if the results are validated
        if validate_vectorization_result(&result) {
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
    model: &str,
    prompts: Vec<String>,
    vector: &mut Vector<String>,
    client: Client<C>
) -> Result<(), Error>
where
    C: Config + Send + Sync + 'static,
{
    // get data from the struct
    let text: String = vector.get_data().clone();

    let shared_client: Arc<Client<C>> = Arc::new(client);
    let shared_text: Arc<String> = Arc::new(text);
    let shared_model: Arc<String> = Arc::new(model.to_string());

    // collect all tasks for concurrent execution
    let mut tasks = Vec::new();
    for (index, prompt) in prompts.into_iter().enumerate() {
        let shared_client: Arc<Client<C>> = shared_client.clone();
        let shared_text: Arc<String> = shared_text.clone();
        let shared_model: Arc<String> = shared_model.clone();

        let task = tokio::spawn(async move {
            let subvector = vectorize_string_single_prompt(
                shared_client.as_ref(),
                shared_model.as_ref(),
                shared_text.as_ref(),
                prompt
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
