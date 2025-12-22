# Quantum PhotoHub - HDR Pipeline (Initial Steps)

This project simulates the initial stages of an HDR processing pipeline for bracketed images, without a backend API yet. It reads images from disk (simulating upload), validates and displays them (via an HTML report), extracts metadata, normalizes and aligns the images, and produces initial HDR results using two techniques:

- Exposure Fusion (Mertens)
- Classic HDR (Debevec) + Tone Mapping (Reinhard)

## Quick Start

1) Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2) Run the pipeline on a folder containing a single bracket set (3–5 images of the same scene, different exposures):

```bash
python main.py --input "D:\path\to\your\bracket_folder" --output ".\output"
```

3) Open the generated HTML report:

- `output\report.html`

This report shows:
- The input images in order (dark → bright) with key EXIF and validation checks
- Normalization and alignment details
- HDR results for Mertens and Debevec+Reinhard
- Timings and basic diagnostics

## Notes

- EXIF orientation is applied automatically when loading images.
- Exposure order is determined by EXIF `ExposureTime` when available; otherwise a best-effort fallback is used.
- Normalization resizes all images to the smallest common width/height, converts to RGB, and scales to float `[0, 1]`.
- Alignment uses MTB (Median Threshold Bitmap) by default; ORB-based alignment is available via `--aligner orb`.

## Command Line

```bash
python main.py --input "<folder_with_bracket_images>" --output "<output_folder>" [--aligner mtb|orb]
```

- `--aligner` defaults to `mtb`. Use `orb` for more robust but slower alignment.

## Folder Expectations

- Place a single bracketed set (same scene, different exposures) in the input folder. Supported formats include JPG/JPEG/PNG/TIF/TIFF.

## Future Steps

- Group multiple bracket sets by EXIF time.
- Add UI/API-based upload.
- Add scene classification and ML-based enhancements.
- Add more tone-mapping options and parameter tuning.


