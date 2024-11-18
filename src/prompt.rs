use std::io::Read;

use anyhow::{Error, Result};

/// this reads prompts from files, and it loads each prompt
/// into the memory for generations. 
pub fn load_prompts(prompt_file_path: &str) -> Result<Vec<String>, Error> {
	let mut prompts: Vec<String> = Vec::new();
	// get all prompt files
	let files: Vec<_> = std::fs::read_dir(prompt_file_path)?
		.filter_map(Result::ok)
		.filter(
			|entry| 
			entry.file_name().to_string_lossy().ends_with(".prompt")
		)
		.collect();
	
	for file in files {
		let mut prompt = String::new();
		let mut prompt_file = std::fs::File::open(
			file.path()
		)?;
		// save the prompt content to the buffer
		prompt_file.read_to_string(&mut prompt)?;
		
		prompts.push(prompt);
	}
	
	Ok(prompts)
}