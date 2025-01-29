use dim_rs::{prelude::*, vectorization::ModelParameters};
use image::DynamicImage;
use tokio;
use anyhow::{Error, Result};
use async_openai::{Client, config::OpenAIConfig};

/// 1. provide examples on how to vectorize image and text
/// 2. provide a real use case of why this method is useful
///     - benchmark
///     - comparison with the traditional embedding models
///         - search speed
///         - memory usage
///         - effectiveness

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Load image
    let image_path: &str = "./examples/images/54e2c8ea-58ef-4871-ae3f-75eabd9a2c6c.jpg";
    let test_image: DynamicImage = image::open(image_path).unwrap();

    // Create a Vector object from the image
    let mut vector: Vector<DynamicImage> = Vector::from_image(test_image);

    // Initialize client
    let client: Client<OpenAIConfig> = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("http://192.168.0.101:11434/v1") // comment this out if you use OpenAI instead of Ollama
            .with_api_key("your_api_key")
    );

    // Initialize prompts
    let prompts: Vec<String> = vec![
        "output in json. Rate the image's offensiveness from 0.0 to 10.0. {'offensiveness': your score}".to_string(),
        "output in json. Rate the image's friendliness from 0.0 to 10.0. {'friendliness': your score}".to_string(),
    ];

    // Initialize model parameters
    let model_parameters = ModelParameters::new(
        "minicpm-v".to_string(), 
        Some(0.7), 
        None
    );

    // Vectorize image
    vectorize_image_concurrently(
        prompts,
        &mut vector, 
        client,
        model_parameters
    ).await?;

    // Print vectorized result
    println!("Vector: {:?}", vector.get_vector());
    println!("Vector Length: {:?}", vector.get_vector().len());

    Ok(())
}