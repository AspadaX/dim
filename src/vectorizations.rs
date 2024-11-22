use std::sync::Arc;

use anyhow::{Error, Result};
use async_openai::{
    config::Config,
    types::{
        ChatCompletionRequestMessageContentPartImageArgs,
        ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, ImageDetail, ImageUrlArgs, ResponseFormat,
    },
    Client,
};
use futures::future::join_all;
use image::DynamicImage;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::{raw_data::utilities::dynamic_image_to_base64, vector::Vector};

/// this file defines codes for vectorizations

pub trait ImageVectorization<C>
where
    C: Send + Sync + 'static + Config,
{
    /// for updating the vector stored in the struct
    fn update_vector(&mut self, vector: Vec<f64>);

    /// for retrieving necessary data.
    /// return an image and prompts
    fn retrieve_data(&self) -> (DynamicImage, Vec<String>);

    /// we will need a way to hold any given vector size and datatypes
    /// generated.
    /// to make the result serializable, the prompt needs to have a
    /// unified output json format:
    /// ```json
    /// {
    /// 	"some_key": usize,
    /// 	"some_other_key": usize,
    /// 	"some_other_key_2"
    /// }
    /// ```

    /// for extracting the leaf values from the respnose we retrieved from
    /// the LLM, as long as it is a valid json.
    fn extract_leaf_values_recursively(&self, value: &Value) -> Vec<Value> {
        match value {
            Value::Object(map) => map
                .values()
                .flat_map(|v| self.extract_leaf_values_recursively(v))
                .collect(),
            Value::Array(arr) => arr
                .iter()
                .flat_map(|v| self.extract_leaf_values_recursively(v))
                .collect(),
            _ => vec![value.clone()],
        }
    }

    /// used to validate if the results are correct
    fn validate_vectorization_result(&self, vector: &Vec<f64>) -> bool;

    /// concurrently generate results for all prompts in the struct,
    /// then gather them, and update them in the vector field of the struct
    async fn vectorize_single_prompt(
        &mut self,
        client: &Client<C>,
        image: &DynamicImage,
        prompt: String,
    ) -> Result<(), Error>
    where
        C: Config + Send + Sync + 'static,
    {
        let base64_image: String = dynamic_image_to_base64(&image)?;
        let image: String = format!("data:image/jpeg;base64,{base64_image}");

        loop {
            let request = CreateChatCompletionRequestArgs::default()
                .model("minicpm-v")
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
            let response = client.chat().create(request).await?;

            // Get the content from the first choice
            let content = &response.choices[0].message.content.clone().unwrap();

            // Parse the content as JSON into ResponseData
            // data validations, if not validated, retry until succeed.
            let result: Vec<f64> = self
                .extract_leaf_values_recursively(&serde_json::from_str(content)?)
                .into_iter()
                .map(|score| score.as_f64().unwrap_or(0.0))
                .collect();
            
            // break the loop if the results are validated
            if self.validate_vectorization_result(&result) {
                self.update_vector(result);
                break;
            }
        }

        Ok(())
    }
}

pub async fn vectorize_image_concurrently<C>(
    vector: &mut Vector<DynamicImage>, 
    client: Client<C>
) -> Result<(), Error>
where
    C: Config + Send + Sync + 'static,
{
    // get data from the struct
    let (image, prompts): (DynamicImage, Vec<String>) = <
        Vector<DynamicImage> as ImageVectorization<C>
    >::retrieve_data(&vector);

    let shared_client: Arc<Client<C>> = Arc::new(client);
    let shared_image: Arc<DynamicImage> = Arc::new(image);
    let shared_vector = Arc::new(
        RwLock::new(vector.clone())
    );

    // collect all tasks for concurrent execution
    let mut tasks = Vec::new();
    for (index, prompt) in prompts.into_iter().enumerate() {
        let shared_client: Arc<Client<C>> = shared_client.clone();
        let shared_image: Arc<DynamicImage> = shared_image.clone();
        let shared_vector = shared_vector
            .clone();
        
        let task = tokio::spawn(async move {
            let mut vector = shared_vector
                .write()
                .await;
            vector.vectorize_single_prompt(
                shared_client.as_ref(), 
                shared_image.as_ref(), 
                prompt
            )
                .await?;
            println!("thread {index} finished vectorization.");

            Ok::<_, Error>(())
        });

        tasks.push(task);
    }

    let _ = join_all(tasks)
        .await;
    
    // update the original vector
    let final_vector: Vec<f64> = {
        let guard = shared_vector.read().await;
        guard.get_vecotr()
    }; // guard is dropped here
    
    vector.overwrite_vector(final_vector);
    
    Ok(())
}
