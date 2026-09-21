# Radiochromic Scanning Tool

Desktop software for radiochromic-film image analysis, scanner flat-field correction, dose-response calibration, uncertainty propagation and traceable result export.

<p align="center">
  <img src="images/main_interface.png" alt="Main application interface" width="700">
</p>

## Main features

- TIFF input with preservation of 8-bit or 16-bit storage depth and Unicode paths.
- Scanner-coordinate flat-field correction.
- Independent R, G and B dose-response calibration.
- Shape-preserving cubic interpolation and rational-model conversion.
- Circular, rectangular, free-form and line measurements.
- Full rational-fit parameter covariance propagation.
- RGB combination using inverse variance, Birge factor or DerSimonian-Laird.
- Automatic film, circular-area and square-area detection.
- Local or global CTR subtraction with membership covariance.
- Centroid, isodose and weighted linear-regression analysis.
- Multi-file processing and versioned CSV export.
- SHA-256 manifests for calibration identity and integrity.

## Installation

Python 3.10 or later is required. Standard Windows Python distributions include `tkinter`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

CUDA acceleration is optional and requires compatible OpenCV/CuPy builds. All scientific functions are available on CPU.

## Recommended workflow

1. Acquire blank scans, calibration films and samples with the same resolution, orientation, bit depth and scanner protocol.
2. Open `Tools → Calibration Wizard` and generate the flat-field and dose-response models.
3. Select the calibration folder and conversion method in Settings.
4. Load images and enable flat-field correction and/or dose conversion.
5. Measure regions manually or with AutoMeasurements.
6. Review invalid or extrapolated regions and export the versioned CSV.

`Auto` dose conversion uses the shape-preserving cubic model inside the calibrated interval and the rational fit outside it. `Spline` never extrapolates. `Fit` uses the rational model over its physical branch.

## Technical documentation

- [Mathematical model and uncertainty](docs/MATHEMATICAL_MODEL.md)
- [Calibration and analysis workflows](docs/WORKFLOWS.md)
- [Data formats, configuration and traceability](docs/DATA_FORMATS.md)
- [Architecture and verification](docs/ARCHITECTURE.md)

## Tests

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app custom_plugins tests main.py
```

Continuous integration runs these checks on Windows with Python 3.10 and 3.12.

## Scientific scope

The program preserves acquisition values and documents its internal uncertainty propagation. Final metrological validity also depends on scanner stability, film orientation and position, post-irradiation time, film lot, reference dose and calibration design. Results must be validated for the specific measurement system and intended use.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
