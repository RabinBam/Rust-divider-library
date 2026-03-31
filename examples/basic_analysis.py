"""
Basic example: Analyzing a single image with Scanner Forensics

Usage:
    python examples/basic_analysis.py --image path/to/image.jpg

This example demonstrates:
1. Loading an image with PIL
2. Initializing the ImageAnalyzer
3. Analyzing patches
4. Interpreting frequency-domain features
5. Detecting AI artifacts
"""

import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """Load image from file and convert to numpy array (float32, RGB)."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        print(f"✓ Loaded image: {image_path}")
        print(f"  Size: {img_array.shape[0]}×{img_array.shape[1]} pixels")
        return img_array
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        sys.exit(1)


def analyze_image(image_array: np.ndarray, filename: str):
    """Analyze image using Scanner Forensics."""
    try:
        # Import here to allow graceful error if not built
        from scanner_forensics import ImageAnalyzer
    except ImportError:
        print("✗ Scanner Forensics library not found!")
        print("  Install with: pip install -e . (or maturin develop --release)")
        sys.exit(1)

    # Initialize analyzer
    print(f"\n📊 Analyzing {filename}...")
    analyzer = ImageAnalyzer(patch_size=256, stride=128)

    # Get patch statistics first
    h, w, c = image_array.shape
    stats = analyzer.get_patch_stats(h, w)
    print(f"  Patch plan: {int(stats['total_patches'])} patches "
          f"({int(stats['num_rows'])}×{int(stats['num_cols'])} grid)")
    print(f"  Coverage: {stats['coverage_percent']:.1f}%")

    # Analyze image
    results = analyzer.analyze_image(image_array, filename, extract_exif=True)

    return results


def interpret_results(results: list):
    """Interpret and visualize analysis results."""
    print(f"\n📈 Analysis Results ({len(results)} patches):")
    print("-" * 80)

    # Calculate statistics
    hf_lf_values = [p.hf_lf_ratio for p in results]
    anomaly_scores = [p.anomaly_score for p in results]
    
    hf_lf_mean = np.mean(hf_lf_values)
    hf_lf_std = np.std(hf_lf_values)
    anomaly_mean = np.mean(anomaly_scores)

    print(f"\n🔍 Frequency Domain Statistics:")
    print(f"  HF/LF Ratio:        {hf_lf_mean:.3f} ± {hf_lf_std:.3f}")
    print(f"    Min: {min(hf_lf_values):.3f}, Max: {max(hf_lf_values):.3f}")
    print(f"  Anomaly Score:      {anomaly_mean:.3f}")
    print(f"  Artifacts Detected: {sum(1 for p in results if p.anomaly_detected)} / {len(results)} patches")

    # AI Detection Heuristic
    print(f"\n🤖 AI Detection Score:")
    if hf_lf_mean < 1.0:
        verdict = "🟢 LIKELY REAL"
        confidence = (1.0 - (hf_lf_mean / 2.0)) * 100
    elif hf_lf_mean < 1.5:
        verdict = "🟡 UNCERTAIN"
        confidence = 50.0
    elif hf_lf_mean < 3.0:
        verdict = "🟠 SUSPICIOUS"
        confidence = (hf_lf_mean / 3.0) * 100
    else:
        verdict = "🔴 LIKELY AI-GENERATED"
        confidence = min((hf_lf_mean / 4.0) * 100, 95.0)

    print(f"  Verdict: {verdict}")
    print(f"  Confidence: {confidence:.1f}%")

    # Top suspicious patches
    print(f"\n⚠️  Top 5 Suspicious Patches:")
    sorted_patches = sorted(results, key=lambda p: p.hf_lf_ratio, reverse=True)
    for i, patch in enumerate(sorted_patches[:5], 1):
        print(f"  {i}. Patch {patch.patch_id} @ ({patch.row_idx}, {patch.col_idx}): "
              f"HF/LF={patch.hf_lf_ratio:.3f}, Anomaly={patch.anomaly_score:.3f}")

    # Band energy analysis
    print(f"\n📊 Energy Distribution (average across patches):")
    low_freq = np.mean([p.low_freq_energy for p in results])
    mid_freq = np.mean([p.mid_freq_energy for p in results])
    high_freq = np.mean([p.high_freq_energy for p in results])
    
    total_energy = low_freq + mid_freq + high_freq
    print(f"  Low:   {(low_freq/total_energy)*100:5.1f}% │{'█' * int((low_freq/total_energy)*20)}")
    print(f"  Mid:   {(mid_freq/total_energy)*100:5.1f}% │{'█' * int((mid_freq/total_energy)*20)}")
    print(f"  High:  {(high_freq/total_energy)*100:5.1f}% │{'█' * int((high_freq/total_energy)*20)}")

    print("-" * 80)


def demonstrate_adversarial(image_array: np.ndarray, filename: str):
    """Optional: Show how degradation affects features."""
    try:
        from scanner_forensics import ImageAnalyzer
        from scanner_forensics.adversarial import (
            apply_jpeg_compression,
            apply_gaussian_blur,
            apply_resize_downup
        )
    except ImportError:
        print("(Skipping adversarial demonstration)")
        return

    print(f"\n🎯 Adversarial Stress Testing (optional):")
    print("-" * 80)

    analyzer = ImageAnalyzer()
    
    # Original
    original = analyzer.analyze_image(image_array, filename)[0]
    print(f"Original:           HF/LF={original.hf_lf_ratio:.3f}")

    # JPEG compression
    degraded_jpeg = apply_jpeg_compression(image_array, quality=50)
    analyzed_jpeg = analyzer.analyze_image(degraded_jpeg, "degraded_jpeg")[0]
    print(f"JPEG (q=50):        HF/LF={analyzed_jpeg.hf_lf_ratio:.3f} (Δ {analyzed_jpeg.hf_lf_ratio - original.hf_lf_ratio:+.3f})")

    # Blur
    degraded_blur = apply_gaussian_blur(image_array, sigma=1.5)
    analyzed_blur = analyzer.analyze_image(degraded_blur, "degraded_blur")[0]
    print(f"Blur (σ=1.5):       HF/LF={analyzed_blur.hf_lf_ratio:.3f} (Δ {analyzed_blur.hf_lf_ratio - original.hf_lf_ratio:+.3f})")

    # Resize
    degraded_resize = apply_resize_downup(image_array, downscale_factor=2)
    analyzed_resize = analyzer.analyze_image(degraded_resize, "degraded_resize")[0]
    print(f"Resize (2×):        HF/LF={analyzed_resize.hf_lf_ratio:.3f} (Δ {analyzed_resize.hf_lf_ratio - original.hf_lf_ratio:+.3f})")

    print("\n💡 Observation: Real images tend to shift HF/LF less under degradation")
    print("    AI images often show larger feature changes (lack of robustness)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image with Scanner Forensics - AI Detection Engine"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to image file (JPG, PNG, WebP)"
    )
    parser.add_argument(
        "--adversarial", "-a",
        action="store_true",
        help="Run adversarial stress tests"
    )

    args = parser.parse_args()

    # Check file exists
    if not Path(args.image).exists():
        print(f"✗ File not found: {args.image}")
        sys.exit(1)

    # Load image
    image_array = load_image(args.image)

    # Analyze
    results = analyze_image(image_array, Path(args.image).name)

    # Interpret
    interpret_results(results)

    # Optional adversarial tests
    if args.adversarial:
        demonstrate_adversarial(image_array, Path(args.image).name)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
