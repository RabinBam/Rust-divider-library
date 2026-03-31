mod divider;
mod error;
mod frequency;
mod metadata;
mod adversarial;

pub use divider::{ImagePatch, PatchConfig, PatchMetadata, SpatialDivider};
pub use frequency::{FrequencyAnalyzer, FrequencyBandStats, ArtifactDetection};
pub use error::{Result, ScannerError};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ndarray::Array3;
use numpy::PyReadonlyArray3;
use std::collections::HashMap;

/// Patch analysis result (Python-callable)
#[pyclass]
pub struct AnalyzedPatch {
    #[pyo3(get)]
    pub patch_id: usize,

    #[pyo3(get)]
    pub row_idx: usize,

    #[pyo3(get)]
    pub col_idx: usize,

    #[pyo3(get)]
    pub absolute_row: usize,

    #[pyo3(get)]
    pub absolute_col: usize,

    #[pyo3(get)]
    pub patch_height: usize,

    #[pyo3(get)]
    pub patch_width: usize,

    // Frequency domain metrics
    #[pyo3(get)]
    pub dc_component: f32,

    #[pyo3(get)]
    pub low_freq_energy: f32,

    #[pyo3(get)]
    pub mid_freq_energy: f32,

    #[pyo3(get)]
    pub high_freq_energy: f32,

    #[pyo3(get)]
    pub hf_lf_ratio: f32,

    #[pyo3(get)]
    pub anomaly_score: f32,

    #[pyo3(get)]
    pub anomaly_detected: bool,

    // Metadata
    #[pyo3(get)]
    pub filename: String,

    #[pyo3(get)]
    pub exif_data: String,
}

#[pymethods]
impl AnalyzedPatch {
    fn __repr__(&self) -> String {
        format!(
            "AnalyzedPatch(id: {}, pos: ({}, {}), hf_lf: {:.3}, anomaly: {:.3})",
            self.patch_id, self.row_idx, self.col_idx, self.hf_lf_ratio, self.anomaly_score
        )
    }

    fn to_dict(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        let _ = dict.set_item("patch_id", self.patch_id);
        let _ = dict.set_item("row_idx", self.row_idx);
        let _ = dict.set_item("col_idx", self.col_idx);
        let _ = dict.set_item("absolute_row", self.absolute_row);
        let _ = dict.set_item("absolute_col", self.absolute_col);
        let _ = dict.set_item("hf_lf_ratio", self.hf_lf_ratio);
        let _ = dict.set_item("anomaly_score", self.anomaly_score);
        let _ = dict.set_item("dc_component", self.dc_component);
        let _ = dict.set_item("high_freq_energy", self.high_freq_energy);
        let _ = dict.set_item("exif_data", &self.exif_data);
        dict.into()
    }
}

/// Image analysis engine (Python-callable)
#[pyclass]
pub struct ImageAnalyzer {
    divider: SpatialDivider,
    analyzer: FrequencyAnalyzer,
    patch_config: PatchConfig,
}

#[pymethods]
impl ImageAnalyzer {
    #[new]
    #[pyo3(signature = (patch_size=256, stride=128))]
    fn new(patch_size: usize, stride: usize) -> PyResult<Self> {
        let config = PatchConfig {
            patch_size,
            stride,
            preserve_aspect: true,
        };

        let divider = SpatialDivider::with_config(config.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let analyzer = FrequencyAnalyzer::with_cache(patch_size);

        Ok(Self {
            divider,
            analyzer,
            patch_config: config,
        })
    }

    /// Analyze image and return list of AnalyzedPatch objects
    /// 
    /// # Arguments
    /// * `image` - numpy array [height, width, channels]
    /// * `filename` - source filename for tracking
    /// 
    /// # Returns
    /// List of AnalyzedPatch objects with frequency analysis
    #[pyo3(signature = (image, filename="unknown.jpg", extract_exif=true))]
    fn analyze_image(
        &self,
        image: PyReadonlyArray3<f32>,
        filename: String,
        extract_exif: bool,
    ) -> PyResult<Vec<AnalyzedPatch>> {
        let img_array = image.as_array().to_owned();

        // Get EXIF data if requested
        let exif_data = if extract_exif {
            metadata::extract_exif(&filename).unwrap_or_default()
        } else {
            String::new()
        };

        // Spatial division
        let patches = self
            .divider
            .divide(&img_array)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut results = Vec::new();

        for (patch_id, patch) in patches.iter().enumerate() {
            // Compute DCT for all channels
            let dct_channels = self
                .analyzer
                .compute_dct_multichannel(&patch.pixels)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Average across channels (for RGB)
            let mut combined_dct = dct_channels[0].clone();
            for i in 1..dct_channels.len() {
                combined_dct = &combined_dct + &dct_channels[i];
            }
            combined_dct = combined_dct.mapv(|x| x / dct_channels.len() as f32);

            // Get frequency band statistics
            let band_stats = self
                .analyzer
                .get_frequency_bands(&combined_dct)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Detect high-frequency artifacts
            let artifact_detection = self
                .analyzer
                .detect_hf_artifacts(&combined_dct, 0.5)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            results.push(AnalyzedPatch {
                patch_id,
                row_idx: patch.metadata.row_idx,
                col_idx: patch.metadata.col_idx,
                absolute_row: patch.metadata.absolute_row,
                absolute_col: patch.metadata.absolute_col,
                patch_height: patch.metadata.height,
                patch_width: patch.metadata.width,
                dc_component: band_stats.dc_component,
                low_freq_energy: band_stats.low_freq_energy,
                mid_freq_energy: band_stats.mid_freq_energy,
                high_freq_energy: band_stats.high_freq_energy,
                hf_lf_ratio: band_stats.hf_lf_ratio,
                anomaly_score: artifact_detection.anomaly_score,
                anomaly_detected: artifact_detection.anomaly_detected,
                filename: filename.clone(),
                exif_data: exif_data.clone(),
            });
        }

        Ok(results)
    }

    /// Get statistics about patch distribution for an image size
    fn get_patch_stats(&self, height: usize, width: usize) -> PyResult<HashMap<String, f32>> {
        let stats = self
            .divider
            .get_patch_stats(height, width)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let mut result = HashMap::new();
        result.insert("total_patches".to_string(), stats.total_patches as f32);
        result.insert("num_rows".to_string(), stats.num_rows as f32);
        result.insert("num_cols".to_string(), stats.num_cols as f32);
        result.insert("overlap_percent".to_string(), stats.overlap_percent);
        result.insert("coverage_percent".to_string(), stats.coverage_percent);

        Ok(result)
    }

    /// Batch analyze multiple images
    #[pyo3(signature = (images, filenames=None))]
    fn analyze_batch(
        &self,
        images: Vec<PyReadonlyArray3<f32>>,
        filenames: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<AnalyzedPatch>>> {
        let mut results = Vec::new();

        for (idx, image) in images.iter().enumerate() {
            let filename = filenames
                .as_ref()
                .and_then(|v| v.get(idx))
                .cloned()
                .unwrap_or_else(|| format!("image_{}.jpg", idx));

            let analyzed = self.analyze_image(image.clone(), filename, true)?;
            results.push(analyzed);
        }

        Ok(results)
    }
}

/// Module initialization for Python
#[pymodule]
#[pyo3(name = "scanner_forensics")]
fn init_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ImageAnalyzer>()?;
    m.add_class::<AnalyzedPatch>()?;
    m.add_submodule(create_adversarial_module(py)?)?;

    Ok(())
}

fn create_adversarial_module(py: Python) -> PyResult<PyModule> {
    let adversarial_mod = PyModule::new(py, "adversarial")?;
    adversarial_mod.add_function(wrap_pyfunction!(apply_jpeg_compression, adversarial_mod)?)?;
    adversarial_mod.add_function(wrap_pyfunction!(apply_gaussian_blur, adversarial_mod)?)?;
    adversarial_mod.add_function(wrap_pyfunction!(apply_resize_downup, adversarial_mod)?)?;
    Ok(adversarial_mod)
}

use pyo3::wrap_pyfunction;

/// JPEG compression degradation
#[pyfunction]
#[pyo3(signature = (image, quality=75))]
fn apply_jpeg_compression(
    image: PyReadonlyArray3<f32>,
    quality: u8,
) -> PyResult<numpy::PyArray3<f32>> {
    let img_array = image.as_array();
    // Simulate JPEG compression artifacts by reducing high frequencies
    let result = adversarial::apply_jpeg_compression(img_array, quality)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(numpy::PyArray3::from_array(
        Python::acquire_gil().python(),
        &result,
    ))
}

/// Gaussian blur degradation
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0))]
fn apply_gaussian_blur(image: PyReadonlyArray3<f32>, sigma: f32) -> PyResult<numpy::PyArray3<f32>> {
    let img_array = image.as_array();
    let result = adversarial::apply_gaussian_blur(img_array, sigma)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(numpy::PyArray3::from_array(
        Python::acquire_gil().python(),
        &result,
    ))
}

/// Resize down then up degradation
#[pyfunction]
#[pyo3(signature = (image, downscale_factor=2))]
fn apply_resize_downup(image: PyReadonlyArray3<f32>, downscale_factor: u32) -> PyResult<numpy::PyArray3<f32>> {
    let img_array = image.as_array();
    let result = adversarial::apply_resize_downup(img_array, downscale_factor)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(numpy::PyArray3::from_array(
        Python::acquire_gil().python(),
        &result,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let _analyzer = ImageAnalyzer::new(256, 128).unwrap();
    }
}
