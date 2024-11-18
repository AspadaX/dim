use anyhow::{Error, Result};
use async_openai::{self, config::OpenAIConfig, Client};

pub fn instantiate_client<C>(
	env_var_name: Option<&str>
) -> Result<Client<OpenAIConfig>, Error> {
	let vars = std::env::vars();
	// determine if the default env var is used
	let mut environment_variable: String = String::new();
	if let Some(env_var_name) = env_var_name {
		environment_variable = env_var_name.to_string();
	} else {
		environment_variable = "OLLAMA_API_BASE".to_string();
	}
	
	// get the api base by using the env var
	let mut api_base: String = String::new();
	for var in vars {
		if var.0 == environment_variable.to_string() {
			api_base = var.1;
		}
	}
	
	println!("using {api_base} as LLM api endpoint...");
	let configuration: OpenAIConfig = OpenAIConfig::new()
		.with_api_key("lm-studio")
		.with_api_base(api_base);
	let client: Client<OpenAIConfig> = Client::with_config(configuration);
	
	Ok(
		client
	)
}
