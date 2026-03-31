use crate::error::{Result, ScannerError};
use ndarray::{Array2, Array3};
use num_complex::Complex;
use std::f32::consts::PI;

/// Frequency analyzer using 2D Discrete Cosine Transform (DCT)
pub struct FrequencyAnalyzer {
    dct_cache: Option<DctCache>,
}

/// Pre-computed DCT cosine values for faster computation
struct DctCache {
    size: usize,
    cos_table: Vec<Vec<f32>>,
}

impl DctCache {
    fn new(size: usize) -> Self {
        let mut cos_table = vec![vec![0.0; size]; size];
        for u in 0..size {
            for x in 0..size {
                let value = ((2.0 * x as f32 + 1.0) * u as f32 * PI) / (2.0 * size as f32);
                cos_table[u][x] = value.cos();
            }
        }
        Self { size, cos_table }
    }

    fn get(&self, u: usize, x: usize) -> f32 {
        self.cos_table[u][x]
    }
}

impl FrequencyAnalyzer {
    /// Create a new frequency analyzer
    pub fn new() -> Self {
        Self { dct_cache: None }
    }

    /// Create with DCT caching for optimal performance
    pub fn with_cache(patch_size: usize) -> Self {
        Self {
            dct_cache: Some(DctCache::new(patch_size)),
        }
    }

    /// Compute 2D DCT for a single channel (grayscale)
    /// 
    /// # Arguments
    /// * `input` - 2D array [height, width]
    /// 
    /// # Returns
    /// 2D DCT coefficients
    pub fn compute_dct_2d(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (height, width) = input.dim();

        if height != width {
            return Err(ScannerError::DctError(
                "DCT requires square patches".to_string(),
            ));
        }

        let size = height;
        let mut output = Array2::<f32>::zeros((size, size));

        // Compute DCT using separable property: 1D row-wise, then 1D column-wise
        // First: DCT on rows
        let mut temp = Array2::<f32>::zeros((size, size));
        for i in 0..size {
            let row = input.row(i).to_owned();
            let dct_row = self.compute_dct_1d(&row.to_vec(), size)?;
            for j in 0..size {
                temp[[i, j]] = dct_row[j];
            }
        }

        // Second: DCT on columns
        for j in 0..size {
            let col = temp.column(j).to_owned().to_vec();
            let dct_col = self.compute_dct_1d(&col, size)?;
            for i in 0..size {
                output[[i, j]] = dct_col[i];
            }
        }

        Ok(output)
    }

    /// Compute 1D DCT
    fn compute_dct_1d(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut output = vec![0.0; n];

        for u in 0..n {
            let mut sum = 0.0;
            for x in 0..n {
                let cos_val = if let Some(ref cache) = self.dct_cache {
                    cache.get(u, x)
                } else {
                    (((2.0 * x as f32 + 1.0) * u as f32 * PI) / (2.0 * n as f32)).cos()
                };
                sum += input[x] * cos_val;
            }

            let alpha = if u == 0 { 1.0 / (n as f32).sqrt() } else { (2.0 / n as f32).sqrt() };
            output[u] = alpha * sum;
        }

        Ok(output)
    }

    /// Extract log-magnitude spectrum (critical for forensics)
    /// 
    /// Returns log(1 + |DCT|) to identify AI fingerprints in high frequencies
    pub fn get_log_magnitude_spectrum(&self, dct: &Array2<f32>) -> Result<Array2<f32>> {
        let log_spectrum = dct.mapv(|x| (1.0 + x.abs()).ln());
        Ok(log_spectrum)
    }

    /// Extract phase spectrum from DCT
    pub fn get_phase_spectrum(&self, dct: &Array2<f32>) -> Result<Array2<f32>> {
        let phase = dct.mapv(|x| x.atan());
        Ok(phase)
    }

    /// Compute frequency band statistics
    /// Divides spectrum into regions and computes energy per region
    pub fn get_frequency_bands(&self, dct: &Array2<f32>) -> Result<FrequencyBandStats> {
        let (height, width) = dct.dim();
        let center_y = height / 2;
        let center_x = width / 2;

        // Define frequency bands from center outward
        let dc_component = dct[[center_y, center_x]];

        // Low frequency (inner 25%)
        let lf_size = center_y / 2;
        let mut lf_energy = 0.0;
        for i in (center_y - lf_size)..(center_y + lf_size) {
            for j in (center_x - lf_size)..(center_x + lf_size) {
                lf_energy += dct[[i, j]].abs();
            }
        }
        lf_energy /= (lf_size * lf_size * 4) as f32;

        // Mid frequency (25-75%)
        let mf_inner = center_y / 4;
        let mf_outer = center_y;
        let mut mf_energy = 0.0;
        let mut mf_count = 0;
        for i in 0..height {
            for j in 0..width {
                let dy = (i as i32 - center_y as i32).abs() as usize;
                let dx = (j as i32 - center_x as i32).abs() as usize;
                let dist = (dy * dy + dx * dx) as f32;
                if dist >= (mf_inner * mf_inner) as f32 && dist < (mf_outer * mf_outer) as f32 {
                    mf_energy += dct[[i, j]].abs();
                    mf_count += 1;
                }
            }
        }
        if mf_count > 0 {
            mf_energy /= mf_count as f32;
        }

        // High frequency (outer 25%)
        let hf_outer = center_y;
        let mut hf_energy = 0.0;
        let mut hf_count = 0;
        for i in 0..height {
            for j in 0..width {
                let dy = (i as i32 - center_y as i32).abs() as usize;
                let dx = (j as i32 - center_x as i32).abs() as usize;
                let dist = (dy * dy + dx * dx) as f32;
                if dist >= (hf_outer * hf_outer) as f32 {
                    hf_energy += dct[[i, j]].abs();
                    hf_count += 1;
                }
            }
        }
        if hf_count > 0 {
            hf_energy /= hf_count as f32;
        }

        Ok(FrequencyBandStats {
            dc_component,
            low_freq_energy: lf_energy,
            mid_freq_energy: mf_energy,
            high_freq_energy: hf_energy,
            hf_lf_ratio: if lf_energy > 0.0 { hf_energy / lf_energy } else { 0.0 },
        })
    }

    /// Multi-channel DCT (grayscale → 1 channel, RGB → 3 channels)
    pub fn compute_dct_multichannel(&self, patch: &Array3<f32>) -> Result<Vec<Array2<f32>>> {
        let (height, width, channels) = patch.dim();

        if height != width {
            return Err(ScannerError::DctError(
                "DCT requires square patches".to_string(),
            ));
        }

        let mut dct_channels = Vec::new();

        for c in 0..channels {
            let channel_data = patch.slice(ndarray::s![.., .., c]);
            let channel_2d = channel_data.into_owned();
            let dct = self.compute_dct_2d(&channel_2d)?;
            dct_channels.push(dct);
        }

        Ok(dct_channels)
    }

    /// Detect high-frequency artifacts (AI fingerprints)
    pub fn detect_hf_artifacts(&self, dct: &Array2<f32>, threshold: f32) -> Result<ArtifactDetection> {
        let (height, width) = dct.dim();
        let total_coeffs = height * width;

        let mut hf_count = 0;
        let mut hf_magnitude_sum = 0.0;

        // High frequency = outer 50% of spectrum
        let threshold_dist = ((height / 2) as f32 * (height / 2) as f32 * 0.5) as usize;

        for i in 0..height {
            for j in 0..width {
                let dy = (i as i32 - (height / 2) as i32).abs() as usize;
                let dx = (j as i32 - (width / 2) as i32).abs() as usize;
                let dist = dy * dy + dx * dx;

                if dist > threshold_dist {
                    let magnitude = dct[[i, j]].abs();
                    if magnitude > threshold {
                        hf_count += 1;
                        hf_magnitude_sum += magnitude;
                    }
                }
            }
        }

        let hf_anomaly_score = hf_count as f32 / total_coeffs as f32;

        Ok(ArtifactDetection {
            anomaly_detected: hf_anomaly_score > 0.1, // 10% threshold
            anomaly_score: hf_anomaly_score,
            artifact_magnitude: if hf_count > 0 { hf_magnitude_sum / hf_count as f32 } else { 0.0 },
            artifact_count: hf_count,
        })
    }
}

/// Frequency band energy statistics
#[derive(Clone, Debug)]
pub struct FrequencyBandStats {
    pub dc_component: f32,
    pub low_freq_energy: f32,
    pub mid_freq_energy: f32,
    pub high_freq_energy: f32,
    pub hf_lf_ratio: f32, // Key indicator for AI generation
}

/// Artifact detection results
#[derive(Clone, Debug)]
pub struct ArtifactDetection {
    pub anomaly_detected: bool,
    pub anomaly_score: f32,
    pub artifact_magnitude: f32,
    pub artifact_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_1d() {
        let analyzer = FrequencyAnalyzer::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = analyzer.compute_dct_1d(&input, 4).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_dct_2d() {
        let analyzer = FrequencyAnalyzer::with_cache(256);
        let input = Array2::<f32>::ones((256, 256));
        let result = analyzer.compute_dct_2d(&input).unwrap();
        assert_eq!(result.dim(), (256, 256));
    }

    #[test]
    fn test_log_magnitude() {
        let analyzer = FrequencyAnalyzer::new();
        let dct = Array2::<f32>::from_elem((4, 4), 1.0);
        let log_spec = analyzer.get_log_magnitude_spectrum(&dct).unwrap();
        assert!(log_spec.iter().all(|&x| x > 0.0));
    }
}
