#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageBuffer, Rgba};
    use dim::prelude::*;

    #[test]
    fn test_from_image() {
        let test_image: DynamicImage = DynamicImage::ImageRgba8(
            ImageBuffer::from_fn(2, 2, |_, _| Rgba([255, 255, 255, 255]))
        );
        let vector_vals: Vec<f32> = vec![];
        let my_vector: Vector<DynamicImage> = Vector::from_image(
            test_image.clone()
        );

        // Verify correctness
        assert_eq!(my_vector.get_vector(), vector_vals);
        assert_eq!(my_vector.get_data(), &test_image);
        assert_eq!(my_vector.get_data_type(), DataType::Image);
        assert_eq!(my_vector.get_dimensionality(), vector_vals.len());
    }

    #[test]
    fn test_from_text() {
        let test_text: String = "Hello, world!".to_string();
        let vector_vals: Vec<f32> = vec![];
        let my_vector: Vector<String> = Vector::from_text(
            test_text.clone()
        );

        // Verify correctness
        assert_eq!(my_vector.get_vector(), vector_vals);
        assert_eq!(my_vector.get_data(), &test_text);
        assert_eq!(my_vector.get_data_type(), DataType::Text);
        assert_eq!(my_vector.get_dimensionality(), vector_vals.len());
    }

    #[test]
    fn test_vector_operations() {
        let test_text: String = "Testing vector ops".to_string();
        let mut my_vector: Vector<String> = Vector::from_text(
            test_text.clone()
        );

        // Test getters
        assert_eq!(my_vector.get_data(), &test_text);
        assert_eq!(my_vector.get_data_type(), DataType::Text);

        // Overwrite vector
        let new_values = vec![1.0, 2.0, 3.0];
        my_vector.overwrite_vector(new_values.clone());
        assert_eq!(my_vector.get_vector(), new_values);
        assert_eq!(my_vector.get_dimensionality(), new_values.len());
    }
}