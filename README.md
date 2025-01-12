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
dim = "0.2.0"
```

## Quick Start

### Vectorize Text

```rust
use dim::prelude::*;
use async_openai;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Create a Vector object from text
    let text = "Hello, world!".to_string();
    let mut vector = Vector::from_text(text);
    
    // Initialize OpenAI client
    let client = async_openai::Client::with_config(
        async_openai::config::OpenAIConfig::new()
            .with_api_key("your_api_key")
    );
    
    // Define prompts for vectorization
    let prompts = vec![
        "output in json. Rate the text's offensiveness from 0.0 to 10.0.".to_string(),
        "output in json. Rate the text's friendliness from 0.0 to 10.0.".to_string(),
    ];

    // Vectorize text
    vectorize_string_concurrently("gpt-4-vision-preview", prompts, &mut vector, client).await?;
    
    println!("Vector: {:?}", vector.get_vector());
    Ok(())
}
```
The result should be something like this:
```rust
Vector: [0.0, 10.0]
```
Notice that each prompt generates a value between 0.0 and 10.0. The final vector is a combination of these values.

### Vectorize Images

```rust
use dim::prelude::*;
use image::DynamicImage;
use async_openai;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Load and create vector from image
    let image = image::open("path/to/image.jpg")?;
    let mut vector = Vector::from_image(image);
    
    // Initialize client and prompts
    let client = async_openai::Client::new();
    let prompts = vec![
        "output in json. Rate the image's brightness from 0.0 to 10.0.".to_string(),
        "output in json. Rate the image's complexity from 0.0 to 10.0.".to_string(),
    ];

    // Vectorize image
    vectorize_image_concurrently("gpt-4-vision-preview", prompts, &mut vector, client).await?;
    
    println!("Vector: {:?}", vector.get_vector());
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