use crate::error::{Result, ScannerError};
use ndarray::{Array3, s};
use image::{ImageBuffer, Rgba};

/// Apply JPEG compression simulation (reduces high-frequency components)
pub fn apply_jpeg_compression(
    image: &ndarray::ArrayView3<f32>,
    quality: u8,
) -> Result<Array3<f32>> {
    let (height, width, channels) = image.dim();

    if channels != 3 && channels != 4 {
        return Err(ScannerError::ImageError(
            "JPEG compression requires RGB or RGBA".to_string(),
        ));
    }

    let mut result = image.to_owned();
    
    // Simulate JPEG blocking artifacts and high-frequency loss
    let block_size = 8;
    let quality_factor = (100 - quality.min(99)) as f32 / 100.0;

    for y in (0..height).step_by(block_size) {
        for x in (0..width).step_by(block_size) {
            let y_end = (y + block_size).min(height);
            let x_end = (x + block_size).min(width);

            for c in 0..channels {
                let mut sum = 0.0;
                let mut count = 0;

                // Calculate mean for block
                for i in y..y_end {
                    for j in x..x_end {
                        sum += result[[i, j, c]];
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let noise = quality_factor * 10.0; // Amplitude scales with quality loss

                // Apply block averaging and noise
                for i in y..y_end {
                    for j in x..x_end {
                        result[[i, j, c]] = mean + (rand::random::<f32>() - 0.5) * noise;
                        result[[i, j, c]] = result[[i, j, c]].clamp(0.0, 255.0);
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Apply Gaussian blur degradation
pub fn apply_gaussian_blur(
    image: &ndarray::ArrayView3<f32>,
    sigma: f32,
) -> Result<Array3<f32>> {
    let (height, width, channels) = image.dim();
    let mut result = Array3::zeros((height, width, channels));

    let kernel_size = ((sigma * 3.0).ceil() as usize * 2 + 1).min(21);
    let kernel_size_half = kernel_size / 2;

    // Pre-compute Gaussian kernel
    let mut kernel = vec![0.0; kernel_size];
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = (i as f32 - kernel_size_half as f32).abs();
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Apply separable convolution
    for c in 0..channels {
        let mut temp = Array3::zeros((height, width, 1));

        // Horizontal pass
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for k in 0..kernel_size {
                    let kx = x as i32 + k as i32 - kernel_size_half as i32;
                    if kx >= 0 && kx < width as i32 {
                        acc += image[[y, kx as usize, c]] * kernel[k];
                    }
                }
                temp[[y, x, 0]] = acc;
            }
        }

        // Vertical pass
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for k in 0..kernel_size {
                    let ky = y as i32 + k as i32 - kernel_size_half as i32;
                    if ky >= 0 && ky < height as i32 {
                        acc += temp[[ky as usize, x, 0]] * kernel[k];
                    }
                }
                result[[y, x, c]] = acc;
            }
        }
    }

    Ok(result)
}

/// Apply downsampling + upsampling degradation
pub fn apply_resize_downup(
    image: &ndarray::ArrayView3<f32>,
    downscale_factor: u32,
) -> Result<Array3<f32>> {
    let (height, width, channels) = image.dim();
    let factor = downscale_factor as usize;

    if factor < 2 {
        return Err(ScannerError::ImageError(
            "Downscale factor must be >= 2".to_string(),
        ));
    }

    // Downsample
    let down_height = (height + factor - 1) / factor;
    let down_width = (width + factor - 1) / factor;
    let mut downsampled = Array3::zeros((down_height, down_width, channels));

    for y in 0..down_height {
        for x in 0..down_width {
            for c in 0..channels {
                let mut sum = 0.0;
                let mut count = 0;

                for dy in 0..factor {
                    for dx in 0..factor {
                        let src_y = y * factor + dy;
                        let src_x = x * factor + dx;
                        if src_y < height && src_x < width {
                            sum += image[[src_y, src_x, c]];
                            count += 1;
                        }
                    }
                }

                downsampled[[y, x, c]] = sum / count as f32;
            }
        }
    }

    // Upsample (nearest neighbor)
    let mut result = Array3::zeros((height, width, channels));
    for y in 0..height {
        for x in 0..width {
            let src_y = (y / factor).min(down_height - 1);
            let src_x = (x / factor).min(down_width - 1);
            for c in 0..channels {
                result[[y, x, c]] = downsampled[[src_y, src_x, c]];
            }
        }
    }

    Ok(result)
}

/// Apply random noise (Gaussian)
pub fn apply_gaussian_noise(
    image: &ndarray::ArrayView3<f32>,
    noise_level: f32,
) -> Result<Array3<f32>> {
    let (height, width, channels) = image.dim();
    let mut result = image.to_owned();

    for i in 0..height {
        for j in 0..width {
            for c in 0..channels {
                let noise = (rand::random::<f32>() - 0.5) * noise_level;
                result[[i, j, c]] = (result[[i, j, c]] + noise).clamp(0.0, 255.0);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_resize_downup() {
        let image = Array3::<f32>::ones((256, 256, 3));
        let result = apply_resize_downup(&image.view(), 2).unwrap();
        assert_eq!(result.dim(), (256, 256, 3));
    }

    #[test]
    fn test_gaussian_blur() {
        let image = Array3::<f32>::ones((256, 256, 3));
        let result = apply_gaussian_blur(&image.view(), 1.0).unwrap();
        assert_eq!(result.dim(), (256, 256, 3));
    }

    #[test]
    fn test_invalid_downscale() {
        let image = Array3::<f32>::ones((256, 256, 3));
        let result = apply_resize_downup(&image.view(), 1);
        assert!(result.is_err());
    }
}
