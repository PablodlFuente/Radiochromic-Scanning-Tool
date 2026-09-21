# Calibration and analysis workflows

## Image acquisition

Use one scanner protocol for blank scans, calibration films and samples:

- identical resolution, storage depth and colour mode;
- identical film orientation and scanner-bed region;
- defined warm-up and post-irradiation timing;
- lossless TIFF storage.

The reader preserves the stored dtype. A dark `uint16` image remains a 16-bit image even when its observed maximum is low.

## Create a calibration

Open `Tools → Calibration Wizard` and select a calibration directory.

### Scanner uniformity

1. Load one or more blank scans acquired with the same protocol.
2. Review per-channel mean, standard deviation, extremes and spatial map.
3. Apply normalization to create `field_flattening.npz`.

All blanks must have the same geometry and dtype and contain finite positive values. Their average is saved as `master_flat.tif` in the calibration directory.

### Dose response

1. Load the calibration TIFF files.
2. Confirm the dose assigned to every image, including dose zero when measured.
3. Define and measure the film ROIs.
4. Review included and excluded R, G and B points.
5. Inspect the rational model (solid line) and PCHIP spline (dashed line) shown together in the fit window.
6. Choose `Save Calibration` from either plot view to save both models.

Saving writes the rational parameters and the exact spline knots. Input rows may be in any order. Repeated doses are averaged for the spline. A non-monotonic channel is rejected with an in-window diagnostic; review or exclude the responsible calibration points rather than silently forcing monotonicity.

Use `Calibration → Modify Calibration` to open the fit editor directly. It restores measured points, saved per-channel exclusions and the recorded bit depth. Click a point marker to exclude it or restore a hollow marker; clicks outside the fixed screen-space selection radius do not alter calibration data. `Restore All Excluded Points` re-includes every point before recalculating. Closing after any edit asks whether both models should be saved.

## Apply dose conversion

1. Select the calibration directory in Settings.
2. Select a conversion method.
3. Load the image.
4. Enable flat-field and dose conversion as required.

| Method | Inside calibrated range | Outside calibrated range |
|---|---|---|
| `Auto` | Shape-preserving cubic interpolation | Rational fit, marked as extrapolated |
| `Spline` | Shape-preserving cubic interpolation | Invalid (`NaN`) |
| `Fit` | Rational fit | Rational fit, marked as extrapolated |

`Auto` is the default. A legacy calibration without a verified spline artifact falls back to `Fit` and displays a warning. Re-save that calibration to enable spline interpolation.

The application verifies manifest hashes before using recorded artifacts. A dose model also records the flat-field used to create it. Replacing the flat requires a compatible dose calibration.

Spline interpolation does not have a compact parameter covariance model. Measurements made in `Spline` or `Auto` report the finite within-ROI standard uncertainty and mark the uncertainty scope as `roi_repeatability_only`; they do not borrow rational-fit covariance for a potentially mixed ROI. `Fit` adds rational-model covariance when a valid covariance matrix is available. The scope is captured with each result and exported.

## Manual measurements

Select a circular, rectangular or line ROI. The panel reports channel means, spatial standard deviations, standard uncertainties, valid-pixel counts and the combined estimate when available.

Histogram display uses at most 1000 deterministic samples; ROI statistics use all valid pixels. Preview binning changes only the displayed image. Measurements and coordinates remain at source resolution.

## AutoMeasurements

1. Add one or more TIFF files.
2. Open the gear button and choose `Circles`, `Squares` or `Circles and squares`; adjust RC, circle and square detection parameters if required.
3. Run detection.
4. Inspect contours and measurements before export.

Automatic detection recognizes circular and square measurement areas. Square detection has independent minimum and maximum side limits. Detected circles are grouped into rows by Y coordinate and ordered by X; `C{row}{column}` expresses that position, while detected squares use `S{index}`. `Add RC` and `Add Measurement Area` open a vertical popover below the pressed button with `Draw rectangle`, `Draw circle` and `Draw custom`. Circles and rectangles are defined by dragging. A custom polygon is defined by successive clicks and completed with Space or right-click. Every measurement area must be completely contained by its assigned RC. Dragging uses a lightweight cyan canvas preview and recalculates the image only after release. An invalid placement is reverted when released. Names must be unique among RCs and among measurement areas in the same RC. Completely invalid regions are not stored as zero-dose measurements.

Right-click a measured circle to open its Tk-owned 3D dose or intensity window. Image-specific results and overlays are cleared before another image is displayed; batch navigation then restores the snapshot belonging to the requested file.

## CTR controls

Mark one or more circles as CTR controls for their film. A global CTR applies to every film in the active image. Original full-precision values are retained, so enabling, disabling or changing a control never accumulates subtraction.

Invalid controls are excluded. When a measured circle belongs to the control mean, its covariance with that mean is included. A single control subtracted from itself is exactly $0\pm0$.

With CTR subtraction active, `Channel dose (raw)` remains the uncorrected per-channel measurement and `Average - CTR` is the combined estimate after control subtraction. A negative corrected value is valid when the selected control is larger than the sample; it is retained rather than clipped to zero.

## Analysis Tools

The plugin provides geometric-versus-dose centroid comparison, four centroid methods, radial isodose contours, introduced-value association and weighted linear regression. The isodose overlay strongly smooths the dose region, forms a robust azimuthal dose profile and draws up to five concentric levels around the calculated dosimetric centroid. This view intentionally represents the radial component of the field instead of scanner-grain microcontours. `Plot Dose vs Value` opens an application-owned Tk figure rather than relying on Matplotlib's global show loop.

## Export

Use `Export CSV` after reviewing the measurements. Numerical values are exported without TreeView rounding. Date, calibration identity, units, method and CTR context come from each measurement's captured provenance.

## Updates and plugins

Disabling a plugin runs its teardown function and removes overlays and references. The integrated updater checks only published GitHub Releases. A newer commit without a release is not offered as an update. The packaged application downloads the release executable, verifies its published size and SHA-256 digest when GitHub provides one, replaces the executable after shutdown and restarts it. Git is not required.
