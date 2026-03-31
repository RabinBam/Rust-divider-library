use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScannerError {
    #[error("Image processing failed: {0}")]
    ImageError(String),

    #[error("Invalid patch configuration: {0}")]
    InvalidPatchConfig(String),

    #[error("DCT computation failed: {0}")]
    DctError(String),

    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    #[error("EXIF extraction failed: {0}")]
    ExifError(String),

    #[error("Invalid input dimensions: {0}")]
    DimensionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image format error: {0}")]
    ImageFormatError(#[from] image::ImageError),
}

pub type Result<T> = std::result::Result<T, ScannerError>;
