mod llm;
mod prompt;
mod vector;
mod raw_data;
mod vectorizations;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use anyhow::Error;
    use image::DynamicImage;
    use llm::instantiate_client;
    use prompt::load_prompts;
    use vectorizations::ImageVectorization;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
    
    #[test]
    fn generate_one_prompt() -> Result<(), Error> {
    	let prompts: Vec<String> = load_prompts(
     		"/Users/xinyubao/Documents/aesthetic-prototype/prompts_clothes"
     	)?;
	    let image: DynamicImage = image::ImageReader::open(
			"/Users/xinyubao/Downloads/clothing-dataset/images/0a3e62e3-fac5-4648-9da2-f6bc4074ee31.jpg"
		)?
			.decode()?;
		let client = instantiate_client(None)?;
		
    	let vector: vector::Vector<DynamicImage> = vector::Vector::new(
     		4, 
       		vec![
         		"accessory_features".to_string(),
           		"collar_attributes".to_string()
         	], 
         	prompts, 
          	2, 
           	image
     	);
     
     	assert_eq!(vector.vectorize_single_prompt(client, image, prompt), vector);
     
     	Ok(())
    }
}
