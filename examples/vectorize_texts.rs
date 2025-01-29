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
