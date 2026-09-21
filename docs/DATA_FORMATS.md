# Data formats, configuration and traceability

## Calibration directory

Each calibration is stored under `calibration_data/<name>/`.

| File | Content |
|---|---|
| `calibration_data.csv` | Dose, intensity statistics and calibration-point provenance. |
| `fit_parameters.csv` | Rational parameters, uncertainties, covariance, dose interval and fit method for each channel. |
| `spline_calibration.npz` | Exact dose/intensity knots and bit depth used to reconstruct the cubic models. |
| `spline_points.csv` | Optional sampled curves for external inspection; not used for conversion. |
| `field_flattening.npz` | Normalized flat, statistics, date, number of blanks and geometry. |
| `master_flat.tif` | Averaged blank image in the calibration directory. |
| `calibration_manifest.json` | SHA-256 identity, artifact hashes, sources and software environment. |

Spline knots are sorted by dose. Replicates at the same dose are averaged before strict monotonicity is evaluated. The stored model is a PCHIP shape-preserving piecewise cubic curve. It avoids the overshoot of an unconstrained natural cubic spline and remains restricted to the measured interval.

`calibration_id` is the SHA-256 digest of the ordered artifact inventory. Changing a recorded artifact changes the identity or causes verification to fail. The manifest and numerical artifacts are written through atomic file replacement.

## Result CSV, schema 3

Every row has `schema_version = 3`. Provenance is captured when the measurement is made and belongs to that result. The complete export is validated before the destination is atomically replaced.

| Column | Meaning |
|---|---|
| `Filename` | Source image filename. |
| `Date` | Metadata date or user-entered measurement date. |
| `Film`, `Circle` | Film and ROI identifiers. |
| `doses_per_channel` | R, G and B dose estimates, or the available channel value. |
| `STD_doses_per_channel` | Spatial standard deviation by channel. |
| `average` | Combined estimate, including CTR correction when active. |
| `standard_uncertainty_average` | Standard uncertainty of `average`. |
| `expanded_uncertainty_k1.96` | $1.96u(\mathrm{average})$; approximate normal coverage. |
| `pixel_count` | Pixels included by ROI geometry. |
| `valid_pixel_counts` | Finite pixels by channel after domain validation. |
| `uncertainty_calculation_method` | Channel-combination method. |
| `channel_weights` | Normalized R, G and B weights. |
| `calibration_id` | Calibration artifact identity. |
| `calibration_integrity` | `verified`, `unverified` or `unknown`. |
| `units` | `Gy` or `scanner_intensity`. |
| `conversion_method` | `auto`, `spline`, `fit` or `not_applied` captured at measurement time. |
| `measurement_status` | `valid`, `partial_pixels`, `partial_channels`, `uncertainty_unavailable` or `invalid`. |
| `x`, `y`, `radius` | ROI geometry in source-image pixels. |
| `dose_correction_factor` | Multiplicative factor applied to the measurement. |
| `flat_applied` | Whether scanner flat-field correction was active. |
| `ctr_context` | JSON description of the applied control. |

Multi-channel vectors are serialized inside quoted CSV fields. `expanded_uncertainty_k1.96` is not a substitute for an experimentally justified coverage factor.

## Configuration

`rc_config.json` is local application state in the repository root and is not tracked by Git. Missing keys receive defaults and saves use atomic replacement.

| Key | Default | Effect |
|---|---:|---|
| `calibration_folder` | `default` | Active calibration subdirectory. |
| `calibration_conversion_method` | `auto` | `auto`, `spline` or `fit` conversion. |
| `uncertainty_estimation_method` | `dersimonian_laird` | RGB combination. |
| `allow_flat_field_resize` | `false` | Explicitly allows flat interpolation to another geometry. |
| `use_multithreading` | `true` | Enables compatible background work. |
| `num_threads` | available CPUs | Worker count. |
| `check_updates_on_startup` | `true` | Checks for updates after startup. |

## Image precision

- `read_image_unchanged` decodes without reducing storage depth and accepts Unicode paths on Windows.
- Storage bit depth is derived from the NumPy dtype, not the observed maximum.
- `write_tiff_unchanged` preserves integer dtype and the API's RGB order.
- Display scaling and preview binning do not change scientific arrays.
- Integral images are computed only where their numerical precision is appropriate.

## Logs and user-visible errors

Each run writes a timestamped file under `logs/`; the most recent ten are retained. Expected workflow failures are shown in dialogs or an in-window status label. Unhandled Tk callback failures are also displayed in the GUI while the traceback remains in the log.
