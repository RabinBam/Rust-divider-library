use crate::error::{Result, ScannerError};
use serde_json::{json, Value};
use std::fs::File;
use std::path::Path;

/// Extract EXIF metadata from image file
pub fn extract_exif(file_path: &str) -> Result<String> {
    let path = Path::new(file_path);

    if !path.exists() {
        return Ok(String::new()); // Return empty if file doesn't exist
    }

    let file = File::open(path)?;
    let mut bufreader = std::io::BufReader::new(&file);

    let exif_data = exif::Reader::new()
        .read_from_container(&mut bufreader)
        .map_err(|e| ScannerError::ExifError(e.to_string()))?;

    let mut metadata = json!({});

    // Extract common EXIF fields
    for field in exif_data.fields() {
        let tag_name = format!("{:?}", field.tag);
        let value_str = format!("{}", field.value.display_as(field.tag));

        metadata[&tag_name] = Value::String(value_str);
    }

    Ok(metadata.to_string())
}

/// Extract image file properties (size, format, etc.)
pub fn extract_file_properties(file_path: &str) -> Result<Value> {
    let path = Path::new(file_path);
    let image = image::open(path)?;

    let metadata = json!({
        "width": image.width(),
        "height": image.height(),
        "color_type": format!("{:?}", image.color()),
        "file_size": std::fs::metadata(path)?.len(),
    });

    Ok(metadata)
}

/// Combine EXIF and file metadata
pub fn get_full_metadata(file_path: &str) -> Result<Value> {
    let exif = extract_exif(file_path).unwrap_or_default();
    let properties = extract_file_properties(file_path)?;

    let combined = json!({
        "exif": exif,
        "properties": properties,
        "file_path": file_path,
    });

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_nonexistent_file() {
        let result = extract_exif("nonexistent.jpg");
        assert!(result.is_ok()); // Should return empty without error
    }
}
