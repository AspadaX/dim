use anyhow::{Error, Result};
use base64::prelude::*;
use image::{self, DynamicImage};

pub fn dynamic_image_to_base64(
	image: &DynamicImage
) -> Result<String, Error> {
	let mut raw_image_bytes: Vec<u8> = Vec::new();
	image.write_to(
	    &mut std::io::Cursor::new(&mut raw_image_bytes),
	    image::ImageFormat::Png,
	)?;
	let base64_image: String = BASE64_STANDARD.encode(
		raw_image_bytes
	);
	
	Ok(base64_image)
}