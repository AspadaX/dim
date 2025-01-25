use dim_rs::prelude::*;
use tokio;
use anyhow::{Error, Result};
use async_openai;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Load text
    let test_text: String = "Hi, this is dim. I am here to vectorize whatever your want."
        .to_string();
    
    // Create a Vector object from the image
    let mut vector: Vector<String> = Vector::from_text(
        test_text
    );
    
    // Initialize client
    let client: async_openai::Client<async_openai::config::OpenAIConfig> = async_openai::Client::with_config(
        async_openai::config::OpenAIConfig::new()
            .with_api_base("http://192.168.0.101:11434/v1") // comment this out if you use OpenAI instead of Ollama
            .with_api_key("your_api_key")
    );
    
    // Initialize prompts
    let prompts: Vec<String> = vec![
        "output in json. Rate the text's offensiveness from 0.0 to 10.0. {'offensiveness': your score}".to_string(),
        "output in json. Rate the text's friendliness from 0.0 to 10.0. {'friendliness': your score}".to_string(),
    ];

    // Vectorize image
    vectorize_string_concurrently(
        "minicpm-v",
        prompts,
        &mut vector, 
        client
    ).await?;

    // Print vectorized image
    println!("{:?}", vector.get_vector());
    
    Ok(())
}
