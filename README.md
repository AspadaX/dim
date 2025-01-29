# dim

`dim` is a Rust library for flexible and extensible vectorization of different types of data (images, text, etc.) using Large Language Models (LLMs). It allows concurrent processing of multiple prompts to generate meaningful vector representations.

## Features

- Support for multiple data types (Image, and Text for now. Other data formats in the future)
- Concurrent vectorization using multiple prompts
- Compatible with OpenAI API format. You may use Ollama API or so as a drop-in replacement 
- Flexible vector dimension control through prompt design
- Built-in validation for vectorization results

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
dim-rs = "0.2.0"
```

## Quick Start

### Vectorize Text

```rust
use dim_rs::{prelude::*, vectorization::ModelParameters};
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
        "Score the sentiment intensity of the text from 1 (extremely negative) to 9 (extremely positive). Consider emotional language, tone, and context. Format your response exactly like this example: {'sentiment_score': 7}".to_string(),
        "Rate the formality of the text from 1 (highly informal, slang-heavy) to 9 (highly formal, academic/professional). Format your response exactly like this example: {'formality_score': 4}".to_string(),
        "Assess the emotional intensity of the text from 1 (neutral/clinical) to 9 (highly emotional, passionate, or provocative). Format your response exactly like this example: {'emotional_score': 8}".to_string(),
        "Score how subjective the text is from 1 (purely factual/objective) to 9 (heavily opinionated/subjective). Format your response exactly like this example: {'subjectivity_score': 6}".to_string(),
        "Rate the linguistic complexity of the text from 1 (simple vocabulary/short sentences) to 9 (dense jargon/long, intricate sentences). Format your response exactly like this example: {'complexity_score': 3}".to_string(),
        "Score the dominant intent: 1-3 (informative/educational), 4-6 (persuasive/argumentative), 7-9 (narrative/storytelling). Format your response exactly like this example: {'intent_score': 5}".to_string(),
        "Rate how urgent or time-sensitive the text feels from 1 (no urgency) to 9 (immediate action required). Format your response exactly like this example: {'urgency_score': 2}".to_string(),
        "Score the specificity of details from 1 (vague/abstract) to 9 (highly specific/concrete examples). Format your response exactly like this example: {'specificity_score': 7}".to_string(),
        "Rate the politeness of the tone from 1 (rude/confrontational) to 9 (extremely polite/deferential). Format your response exactly like this example: {'politeness_score': 8}".to_string(),
        "Categorize the text's primary domain: 1-3 (technical/scientific), 4-6 (casual/everyday), 7-9 (artistic/creative). Format your response exactly like this example: {'domain_score': 4}".to_string(),
    ];

    // Vectorize image
    let model_parameters = ModelParameters::new("minicpm-v".to_string(), None, None);
    vectorize_string_concurrently(
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
```
The result should be something like this:
```rust
Vector: [7, 4, 8, 6, 3, 5, 2, 7, 8, 4]
```
Notice that each prompt generates a value between 1 and 9. The final vector is a combination of these values.

### Vectorize Images

```rust
use dim_rs::{prelude::*, vectorization::ModelParameters};
use image::DynamicImage;
use tokio;
use anyhow::{Error, Result};
use async_openai::{Client, config::OpenAIConfig};

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
```
Once again, the result should be something like this:
```rust
Vector: [0.0, 10.0]
```
Notice that each prompt generates a value between 0.0 and 10.0. The final vector is a combination of these values.

## How It Works

1. The library takes your data (text/image) and creates a `Vector` object
2. You provide multiple prompts that will be used to analyze different aspects of the data
3. The prompts are processed concurrently using the specified LLM
4. Results are combined into a single vector representation
5. The dimensionality of the final vector is determined by the number of prompts and their specified outputs

## Configuration

- Works with OpenAI API style. Also, this project uses `async_openai` for API calls. 
- Customize API endpoint using:
```rust
.with_api_base("your_api_endpoint")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.