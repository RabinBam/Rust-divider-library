use crate::error::{Result, ScannerError};
use ndarray::{s, Array3};
use rayon::prelude::*;

/// Configuration for patch extraction
#[derive(Clone, Debug)]
pub struct PatchConfig {
    /// Patch width (default: 256)
    pub patch_size: usize,
    /// Stride/hop size (default: 128 for 50% overlap)
    pub stride: usize,
    /// Preserve original aspect ratio
    pub preserve_aspect: bool,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_size: 256,
            stride: 128,
            preserve_aspect: true,
        }
    }
}

/// Patch metadata container
#[derive(Clone, Debug)]
pub struct PatchMetadata {
    pub row_idx: usize,
    pub col_idx: usize,
    pub absolute_row: usize,
    pub absolute_col: usize,
    pub height: usize,
    pub width: usize,
}

/// Single image patch with spatial context
pub struct ImagePatch {
    pub pixels: Array3<f32>,          // [height, width, channels]
    pub metadata: PatchMetadata,
    pub confidence: f32,               // For quality assessment
}

/// Spatial divider for image patching
pub struct SpatialDivider {
    config: PatchConfig,
}

impl SpatialDivider {
    /// Create a new spatial divider with default config
    pub fn new() -> Self {
        Self {
            config: PatchConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PatchConfig) -> Result<Self> {
        if config.patch_size == 0 || config.stride == 0 {
            return Err(ScannerError::InvalidPatchConfig(
                "patch_size and stride must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Divide an image into overlapping patches using sliding window
    /// 
    /// # Arguments
    /// * `image` - Input image array [height, width, channels]
    /// 
    /// # Returns
    /// Vector of ImagePatch objects with spatial metadata
    pub fn divide(&self, image: &Array3<f32>) -> Result<Vec<ImagePatch>> {
        let shape = image.dim();
        let (height, width, _channels) = (shape.0, shape.1, shape.2);

        if height < self.config.patch_size || width < self.config.patch_size {
            return Err(ScannerError::DimensionError(format!(
                "Image ({}, {}) smaller than patch size ({})",
                height, width, self.config.patch_size
            )));
        }

        let row_starts = self.compute_patch_starts(height);
        let col_starts = self.compute_patch_starts(width);

        let mut patch_locations = Vec::with_capacity(row_starts.len() * col_starts.len());
        for (row_idx, &row_start) in row_starts.iter().enumerate() {
            for (col_idx, &col_start) in col_starts.iter().enumerate() {
                patch_locations.push((row_idx, col_idx, row_start, col_start));
            }
        }

        patch_locations
            .into_par_iter()
            .map(|(row_idx, col_idx, row_start, col_start)| {
                let patch_pixels = image
                    .slice(s![
                        row_start..row_start + self.config.patch_size,
                        col_start..col_start + self.config.patch_size,
                        ..
                    ])
                    .to_owned();

                Ok(ImagePatch {
                    pixels: patch_pixels,
                    metadata: PatchMetadata {
                        row_idx,
                        col_idx,
                        absolute_row: row_start,
                        absolute_col: col_start,
                        height: self.config.patch_size,
                        width: self.config.patch_size,
                    },
                    confidence: 1.0,
                })
            })
            .collect()
    }

    /// Divide with explicit handling for non-square images
    pub fn divide_adaptive(&self, image: &Array3<f32>) -> Result<Vec<ImagePatch>> {
        let shape = image.dim();
        let (height, width, _channels) = (shape.0, shape.1, shape.2);

        // For very large images, use variable patch sizes
        let adaptive_patch_size = if height > 2048 || width > 2048 {
            512
        } else if height > 1024 || width > 1024 {
            384
        } else {
            256
        };

        let config = PatchConfig {
            patch_size: adaptive_patch_size,
            stride: adaptive_patch_size / 2, // 50% overlap
            preserve_aspect: true,
        };

        let divider = Self::with_config(config)?;
        divider.divide(image)
    }

    fn compute_patch_starts(&self, dimension: usize) -> Vec<usize> {
        let patch = self.config.patch_size;
        let stride = self.config.stride;

        let mut starts = Vec::new();
        let mut current = 0;

        while current + patch <= dimension {
            starts.push(current);
            current += stride;
        }

        // Force coverage of the right/bottom edge for non-divisible dimensions.
        let last_start = dimension - patch;
        if starts.last().copied() != Some(last_start) {
            starts.push(last_start);
        }

        starts
    }

    /// Get statistics about patch distribution
    pub fn get_patch_stats(&self, image_height: usize, image_width: usize) -> Result<PatchStats> {
        if image_height < self.config.patch_size || image_width < self.config.patch_size {
            return Err(ScannerError::DimensionError(
                "Image too small for patching".to_string(),
            ));
        }

        let num_rows = self.compute_patch_starts(image_height).len();
        let num_cols = self.compute_patch_starts(image_width).len();
        let total_patches = num_rows * num_cols;

        // Calculate overlap statistics
        let row_overlap_percent = (1.0 - (self.config.stride as f32 / self.config.patch_size as f32)) * 100.0;

        Ok(PatchStats {
            total_patches,
            num_rows,
            num_cols,
            patch_size: self.config.patch_size,
            stride: self.config.stride,
            overlap_percent: row_overlap_percent,
            coverage_percent: (total_patches as f32 * self.config.patch_size as f32 * self.config.patch_size as f32)
                / (image_height as f32 * image_width as f32) * 100.0,
        })
    }
}

/// Statistics about patch distribution
#[derive(Clone, Debug)]
pub struct PatchStats {
    pub total_patches: usize,
    pub num_rows: usize,
    pub num_cols: usize,
    pub patch_size: usize,
    pub stride: usize,
    pub overlap_percent: f32,
    pub coverage_percent: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_extraction() {
        let divider = SpatialDivider::new();
        let image = Array3::<f32>::ones((512, 512, 3));
        let patches = divider.divide(&image).unwrap();
        
        // With 256 patch size and 128 stride on 512x512: (512-256)/128 + 1 = 3 patches per dimension
        assert_eq!(patches.len(), 9); // 3x3
    }

    #[test]
    fn test_edge_coverage_for_non_divisible_dimensions() {
        let divider = SpatialDivider::new();
        let image = Array3::<f32>::ones((500, 500, 3));
        let patches = divider.divide(&image).unwrap();

        assert_eq!(patches.len(), 9);
        let last_patch = patches
            .iter()
            .find(|p| p.metadata.row_idx == 2 && p.metadata.col_idx == 2)
            .unwrap();

        assert_eq!(last_patch.metadata.absolute_row, 244);
        assert_eq!(last_patch.metadata.absolute_col, 244);
    }

    #[test]
    fn test_patch_shape_is_consistent() {
        let divider = SpatialDivider::new();
        let image = Array3::<f32>::ones((300, 300, 3));
        let patches = divider.divide(&image).unwrap();

        for patch in patches {
            assert_eq!(patch.pixels.dim(), (256, 256, 3));
            assert_eq!(patch.metadata.height, 256);
            assert_eq!(patch.metadata.width, 256);
        }
    }

    #[test]
    fn test_invalid_config() {
        let result = SpatialDivider::with_config(PatchConfig {
            patch_size: 0,
            stride: 128,
            preserve_aspect: true,
        });
        assert!(result.is_err());
    }
}
