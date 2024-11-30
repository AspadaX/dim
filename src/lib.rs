pub mod llm;
pub mod prompt;
pub mod vector;
pub mod raw_data;
pub mod vectorizations;

#[cfg(test)]
mod tests {
    use anyhow::Error;
    use async_openai::{config::OpenAIConfig, Client};
    use image::DynamicImage;
    use llm::instantiate_client;
    use prompt::load_prompts;
    use vectorizations::vectorize_image_concurrently;

    use super::*;

    #[tokio::test]
    async fn test_vectorize_an_image() -> Result<(), Error> {
    	let prompts: Vec<String> = load_prompts(
     		"/Users/xinyubao/Downloads/prompts"
     	)?;
	    let image: DynamicImage = image::ImageReader::open(
			"/Users/xinyubao/Downloads/clothing-dataset/images/0a3e62e3-fac5-4648-9da2-f6bc4074ee31.jpg"
		)?
			.decode()?;
		let client: Client<
		  OpenAIConfig
		> = instantiate_client::<OpenAIConfig>(None)?;
		
    	let mut vector: vector::Vector<DynamicImage> = vector::Vector::new(
     		4, 
       		vec![
         		"accessory_features".to_string(),
           		"collar_attributes".to_string()
         	], 
         	prompts, 
          	2, 
           	image
     	);
     
        vectorize_image_concurrently::<OpenAIConfig>(&mut vector, client)
            .await?;
        
        let new_vector: Vec<f64> = vector.get_vector();
     
     	assert_eq!(
      		new_vector.len(),
        	4
      	);
     
     	Ok(())
    }
}
