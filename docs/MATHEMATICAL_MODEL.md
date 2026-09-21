# Mathematical model and uncertainty

## Notation

- $I_k(x,y)$: scanner intensity for channel $k\in\{R,G,B\}$.
- $D$: absorbed dose.
- $a_k,b_k,c_k$: rational response parameters.
- $N_k$: valid pixels for channel $k$ in an ROI.
- $u(x)$: standard uncertainty of $x$.

Calculations use `float64`. Non-finite data, singularities and values outside an accepted domain are represented as `NaN`, never as zero.

## Scanner flat-field correction

For channel $k$, blank scans are averaged to obtain $B_k(x,y)$. With spatial mean $\bar B_k$, the normalized flat is

$$
F_k(x,y)=\frac{B_k(x,y)}{\bar B_k}.
$$

The corrected image is

$$
I_{k,\mathrm{flat}}(x,y)=\frac{I_k(x,y)}{F_k(x,y)}.
$$

The result is clipped to the storage type and preserves its dtype. Flat and sample must share scanner geometry unless explicit resizing is enabled.

## Rational dose-response model

Each channel is fitted independently:

$$
I_k(D)=a_k+\frac{b_k}{D-c_k},
$$

with $b_k\ge 0$ and $c_k$ constrained below the minimum included dose. At least three finite points are required. Positive ROI spatial deviations are supplied to nonlinear least squares with `absolute_sigma=True`.

The inverse is

$$
D_k(I)=c_k+\frac{b_k}{I-a_k}.
$$

It is undefined near $I=a_k$. The physical branch used by the application satisfies $D>c_k$. Values outside the fitted dose interval are marked as extrapolated.

Example: for $(a,b,c)=(1000,5000,-1)$ and $I=2000$,

$$
D=-1+\frac{5000}{2000-1000}=4.
$$

## Shape-preserving cubic model

Calibration rows are sorted by dose and replicate intensities at the same dose are averaged. The resulting knots $(D_j,I_j)$ must be strictly monotonic in intensity. A PCHIP piecewise cubic Hermite interpolant is constructed:

$$
I_k(D)=S_k(D),\qquad D\in[D_{k,\min},D_{k,\max}].
$$

PCHIP is cubic on each interval and preserves the monotonic shape of the knots. It is used instead of an unconstrained natural cubic spline because it does not introduce overshoot between monotonic calibration points.

For conversion, the monotonic knot order is reversed when necessary and a shape-preserving inverse interpolant is built:

$$
D_k(I)=S_k^{-1}(I),\qquad I\in[I_{k,\min},I_{k,\max}].
$$

No spline extrapolation is performed. In `Auto`, in-range pixels use $S_k^{-1}$ and pixels outside the knot interval use the rational inverse. In `Spline`, outside pixels are invalid. In `Fit`, all physically valid pixels use the rational inverse.

## ROI statistics

For the $N_k$ finite channel values,

$$
\bar D_k=\frac{1}{N_k}\sum_{j=1}^{N_k}D_{k,j},
\qquad
s_k=\sqrt{\frac{1}{N_k}\sum_{j=1}^{N_k}(D_{k,j}-\bar D_k)^2},
$$

and the implemented sampling term is

$$
u_{k,\mathrm{stat}}=\frac{s_k}{\sqrt{N_k}}.
$$

For the rational inverse, the Jacobian with respect to $\theta=(a,b,c)$ is

$$
J(I)=\left(\frac{b}{(I-a)^2},\frac{1}{I-a},1\right).
$$

The Jacobian is averaged over exactly the pixels that produced finite doses. With the complete parameter covariance $C_k$,

$$
u_{k,\mathrm{cal}}^2=\bar J_k C_k\bar J_k^\mathsf{T},
$$

and

$$
u_k=\sqrt{u_{k,\mathrm{stat}}^2+u_{k,\mathrm{cal}}^2}.
$$

The parameter contribution is shared by all pixels and is not divided by $N_k$. If covariance is missing or invalid, it is reported as unavailable (`NaN`). Because a mixed `Auto` ROI can contain both spline and rational values, parameter covariance is conservatively reported as unavailable for both `Spline` and `Auto`; `Fit` uses rational-model covariance propagation.

## Channel combination

Only channels with finite dose and positive finite uncertainty participate.

### Inverse-variance mean

$$
w_k=\frac{1}{u_k^2},\qquad
\bar D=\frac{\sum_k w_kD_k}{\sum_k w_k},\qquad
u(\bar D)=\frac{1}{\sqrt{\sum_k w_k}}.
$$

### Birge factor

For $n$ valid channels,

$$
\chi^2=\sum_k w_k(D_k-\bar D)^2,
\qquad
R_B=\max\left(1,\sqrt{\frac{\chi^2}{n-1}}\right),
$$

and $u_B(\bar D)=R_B/\sqrt{\sum w_k}$.

### DerSimonian-Laird

With fixed-effect weights $w_k$,

$$
Q=\sum_k w_k(D_k-\bar D_F)^2,
$$

$$
C=\sum_k w_k-\frac{\sum_k w_k^2}{\sum_k w_k},
\qquad
\tau^2=\max\left(0,\frac{Q-(n-1)}{C}\right).
$$

Random-effect weights are $w_k^*=1/(u_k^2+\tau^2)$.

## CTR subtraction

For $m$ valid controls,

$$
\bar C=\frac{1}{m}\sum_{j=1}^{m}C_j,
\qquad
u(\bar C)^2=\frac{1}{m^2}\sum_{j=1}^{m}u(C_j)^2.
$$

For an independent measurement $X$,

$$
X'=X-\bar C,
\qquad
u(X')^2=u(X)^2+u(\bar C)^2.
$$

If $X$ is a member of the control mean, $\operatorname{Cov}(X,\bar C)=u(X)^2/m$ and

$$
u(X')^2=u(X)^2+u(\bar C)^2-2\operatorname{Cov}(X,\bar C).
$$

## Weighted regression

For introduced value $x_i$, measured dose $y_i$ and dose uncertainty $u(y_i)$, the analysis minimizes

$$
\sum_i\frac{(y_i-(mx_i+q))^2}{u(y_i)^2}.
$$

The intercept may be fitted or fixed at zero. With absolute uncertainties, covariance is $(X^\mathsf{T}WX)^{-1}$. Without usable uncertainties, unit weights are used and covariance is scaled by residual variance. Insufficient or singular data return `NaN`.

## Scope of the uncertainty budget

The $s/\sqrt N$ term assumes independent pixel fluctuations and does not estimate an effective sample size from spatial autocorrelation. RGB combination does not include an experimentally estimated inter-channel covariance matrix. CTR subtraction includes membership covariance but not every shared scanner, reference-dose or calibration contribution.

Flat-field uncertainty and additional protocol contributions require external evaluation when relevant. The reported uncertainty is therefore the implemented software budget, not a guarantee of complete metrological coverage. Factor 1.96 represents approximate normal coverage.

## References

1. Fritsch, F. N., & Carlson, R. E. (1980). *Monotone Piecewise Cubic Interpolation*. SIAM Journal on Numerical Analysis, 17(2), 238–246. https://doi.org/10.1137/0717021
2. Birge, R. T. (1932). *The Calculation of Errors by the Method of Least Squares*. Physical Review, 40(2), 207–227. https://doi.org/10.1103/PhysRev.40.207
3. DerSimonian, R., & Laird, N. (1986). *Meta-analysis in clinical trials*. Controlled Clinical Trials, 7(3), 177–188. https://doi.org/10.1016/0197-2456(86)90046-2
