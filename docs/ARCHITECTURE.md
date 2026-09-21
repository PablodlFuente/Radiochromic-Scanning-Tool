# Architecture and verification

## Responsibilities

| Component | Responsibility |
|---|---|
| `main.py`, `app/rc_analyzer.py` | Application startup, logging and Tk lifecycle. |
| `app/ui/` | Windows, user events, presentation and workflow coordination. |
| `app/core/image_processor.py` | Image state, corrections, ROI measurements and display preparation. |
| `app/core/dosimetry.py` | Rational response, domain checks, covariance propagation and RGB combination. |
| `app/core/spline_calibration.py` | Monotonic cubic knots, inversion and durable spline storage. |
| `app/core/calibration_manifest.py` | Artifact identity, provenance and integrity verification. |
| `app/calibration/` | Calibration wizard, ROI extraction and dose-response modelling. |
| `app/utils/` | Image I/O, configuration, files and release updating. |
| `app/plugins/` | Plugin discovery, activation, notifications and shutdown. |
| `custom_plugins/auto_measurements/` | Detection, per-file results, CTR processing and export. |
| `custom_plugins/analysis_tools/` | Centroids, isodoses and weighted regression. |
| `tests/` | Numerical, regression and integration verification. |

The analysis pipeline is:

`original image → optional flat-field → optional dose conversion → ROI → channel combination → optional CTR → export`

Preview binning belongs only to display preparation. It does not alter analysis arrays or coordinates.

## State and concurrency

The image processor protects mutable operations with a reentrant lock. Image-load requests carry identifiers so stale completions cannot overwrite newer state. Worker results reach Tk through a queue, and display callbacks compare state revisions before committing results. Geometry dragging does not enqueue full image renders for every pointer event: Tk draws a vector preview and commits one image refresh on release, preventing worker-thread and lock contention.

Loading another image invalidates image-specific plugin results and overlays before any saved batch snapshot is restored. Measurements store their own provenance, and per-file result sets are deep-copied. Export never relabels previous results with the currently selected calibration.

Interactive Matplotlib figures are embedded in Tk-owned windows. Unhandled Tk callback exceptions are written to the detailed log and shown in the graphical interface.

## Verification strategy

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app custom_plugins tests main.py
python -m pip check
```

Numerical tests use analytical examples, known inversions and unit transformations. Integration tests exercise temporary TIFF files, 16-bit storage, calibration manifests, atomic output and isolated Git repositories. The Tk smoke tests use a real event loop, sequentially load images and open a plot window. They are skipped explicitly when no Tk display is available.

Release verification additionally builds the wheel and one-file Windows executable. The executable is started from its distribution directory and must load both bundled plugins while keeping writable configuration and calibration paths beside the executable.

Mocks trigger processing failures and output-replacement errors. They test workflow contracts, not experimental accuracy. The suite does not replace visual acceptance, large-image stress testing, GPU qualification or comparison against independent reference doses.

## Structural boundaries

The directory structure separates numerical code, UI, calibration, utilities and plugins. Some UI controllers remain large and should not receive new numerical logic. New calculation features should be introduced as Tk-independent services with explicit data contracts and tests, then called by the controllers.
