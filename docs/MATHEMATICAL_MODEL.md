# Modelo matemático e incertidumbre

## Convenciones

- $I_k$: intensidad del canal $k\in\{R,G,B\}$.
- $D$: dosis.
- $a_k,b_k,c_k$: parámetros del canal.
- $C_k$: matriz de covarianza $3\times3$ de $(a_k,b_k,c_k)$.
- $N$: número de píxeles válidos de una ROI.
- $u(x)$: incertidumbre estándar de $x$.

Los cálculos se realizan en `float64`. Los píxeles fuera del dominio, las singularidades y los datos no finitos se representan como `NaN` y no se convierten en cero.

## Corrección de uniformidad del escáner

Para cada canal se promedian los blancos adquiridos. Si $B_k(x,y)$ es el blanco promedio y $\bar B_k$ su media espacial, el flat normalizado es

$$
F_k(x,y)=\frac{B_k(x,y)}{\bar B_k}.
$$

La imagen corregida es

$$
I_{k,\mathrm{corr}}(x,y)=\frac{I_k(x,y)}{F_k(x,y)}.
$$

El resultado se recorta al intervalo del tipo de almacenamiento y conserva su `dtype`. El flat y la imagen deben compartir dimensiones: la corrección está definida en coordenadas físicas del escáner. El redimensionado sólo se ejecuta cuando `allow_flat_field_resize` es `true`.

## Curva dosis–respuesta

Cada canal se ajusta de forma independiente al modelo racional

$$
I_k(D)=a_k+\frac{b_k}{D-c_k},
$$

con $b_k\ge 0$ y $c_k$ restringido por debajo de la dosis mínima del ajuste. Se necesitan al menos tres puntos finitos por canal. El punto de dosis cero participa en el ajuste cuando está presente.

El ajuste automático usa mínimos cuadrados no lineales. Cuando existen desviaciones positivas de las ROIs, se emplean como $\sigma_I$ con `absolute_sigma=True`, por lo que la covarianza estimada conserva la escala declarada. La desviación espacial de una ROI de calibración no se divide por $\sqrt N$: los píxeles del escáner tienen correlación espacial y no se asumen réplicas independientes.

La conversión inversa es

$$
D_k(I)=c_k+\frac{b_k}{I-a_k}.
$$

La operación no está definida si $I\simeq a_k$. El intervalo $[D_{k,\min},D_{k,\max}]$ se guarda junto con el ajuste. Por defecto, una dosis inversa fuera de ese intervalo queda en `NaN`. Si se permite extrapolación, la máscara de extrapolación sigue disponible para identificar esos píxeles.

### Ejemplo

Para $(a,b,c)=(1000,5000,-1)$ e intensidad $I=2000$:

$$
D=-1+\frac{5000}{2000-1000}=4.
$$

## Estadística de una ROI

Para cada canal se usan sólo los $N$ píxeles finitos. Se calculan

$$
\bar D_k=\frac{1}{N}\sum_{j=1}^{N}D_{k,j},
\qquad
s_k=\sqrt{\frac{1}{N}\sum_{j=1}^{N}(D_{k,j}-\bar D_k)^2},
$$

y la contribución estadística

$$
u_{k,\mathrm{stat}}=\frac{s_k}{\sqrt N}.
$$

La covarianza del ajuste es una fuente compartida por todos los píxeles. Para $D=c+b/(I-a)$, el jacobiano respecto de $\theta=(a,b,c)$ es

$$
J(I)=\left(\frac{b}{(I-a)^2},\frac{1}{I-a},1\right).
$$

Se promedia el jacobiano de la ROI y se propaga la matriz completa:

$$
u_{k,\mathrm{cal}}^2=\bar J_k C_k\bar J_k^\mathsf{T}.
$$

Esta contribución no se divide por $N$. La incertidumbre estándar total del canal es

$$
u_k=\sqrt{u_{k,\mathrm{stat}}^2+u_{k,\mathrm{cal}}^2}.
$$

Si la covarianza del ajuste no está disponible, la incertidumbre total calibrada se declara no disponible (`NaN`).

## Combinación de canales

Sólo participan canales con $D_k$ y $u_k>0$ finitos.

### Media ponderada por varianza inversa

$$
w_k=\frac{1}{u_k^2},\qquad
\bar D=\frac{\sum_k w_kD_k}{\sum_k w_k},\qquad
u(\bar D)=\frac{1}{\sqrt{\sum_k w_k}}.
$$

`sensitivity_weighted` utiliza esta misma expresión porque $u_k$ ya está expresada en dosis e incorpora la sensibilidad de la curva.

### Factor de Birge

Con $n$ canales válidos,

$$
\chi^2=\sum_k w_k(D_k-\bar D)^2,
\qquad
R_B=\sqrt{\max\left(1,\frac{\chi^2}{n-1}\right)}.
$$

La incertidumbre combinada es $R_B/\sqrt{\sum w_k}$. El factor sólo amplía la incertidumbre cuando la dispersión entre canales excede la esperada.

### DerSimonian–Laird

Se calcula

$$
Q=\sum_k w_k(D_k-\bar D_F)^2,
$$

$$
C=\sum_k w_k-\frac{\sum_k w_k^2}{\sum_k w_k},
\qquad
\tau^2=\max\left(0,\frac{Q-(n-1)}{C}\right).
$$

Los pesos finales son $w_k^*=1/(u_k^2+\tau^2)$. La media y su incertidumbre se calculan con $w_k^*$.

## Sustracción de controles CTR

Para $m$ controles, el valor de control es la media aritmética

$$
\bar C=\frac{1}{m}\sum_i C_i.
$$

La varianza propagada de esa media es $\sum_i u(C_i)^2/m^2$. La varianza observada de la media es $s_C^2/m$. El programa usa la mayor de ambas para cubrir heterogeneidad sin sumar dos veces el mismo ruido:

$$
u(\bar C)^2=\max\left(\frac{\sum_i u(C_i)^2}{m^2},\frac{s_C^2}{m}\right).
$$

Para una medida independiente $X$:

$$
X'=X-\bar C,\qquad
u(X')^2=u(X)^2+u(\bar C)^2.
$$

Si $X$ es uno de los controles incluidos en $\bar C$, se incorpora $\operatorname{Cov}(X,\bar C)=u(X)^2/m$:

$$
u(X')^2=u(X)^2+u(\bar C)^2-2\operatorname{Cov}(X,\bar C).
$$

Un único control restado de sí mismo da exactamente $0\pm0$.

## Regresión del módulo de análisis

Para la relación entre valor introducido $x_i$ y dosis medida $y_i$, el ajuste minimiza

$$
\sum_i\frac{(y_i-(mx_i+q))^2}{u(y_i)^2}.
$$

Puede estimarse $q$ o imponer $q=0$. La covarianza es $(X^\mathsf{T}WX)^{-1}$ cuando las incertidumbres son absolutas. Si no hay incertidumbres utilizables, el ajuste usa pesos unitarios y escala la covarianza por la varianza residual. El $R^2$ mostrado usa sumas de cuadrados ponderadas. Los conjuntos insuficientes o singulares devuelven `NaN`.

## Centroides

El módulo de análisis compara el centro geométrico de cada círculo con un centro estimado en dosis. Están disponibles centroide ponderado, potencia gaussiana, ponderación radial y correlación de fase. Las coordenadas calculadas en una ROI recortada se trasladan al origen real de esa ROI en la imagen. Las isodosis son contornos calculados dentro de cada región.

## Referencias

1. Birge, R. T. (1932). *The Calculation of Errors by the Method of Least Squares*. Physical Review, 40(2), 207–227. https://doi.org/10.1103/PhysRev.40.207
2. DerSimonian, R., & Laird, N. (1986). *Meta-analysis in clinical trials*. Controlled Clinical Trials, 7(3), 177–188. https://doi.org/10.1016/0197-2456(86)90046-2
