# Formatos, configuración y trazabilidad

## Carpeta de calibración

Cada calibración reside bajo `calibration_data/<nombre>/`.

| Archivo | Contenido |
|---|---|
| `calibration_data.csv` | Dosis, estadísticos de intensidad y procedencia de los puntos. |
| `fit_parameters.csv` | $a,b,c$, incertidumbres, covarianza completa, intervalo de dosis, número de puntos y método por canal. |
| `field_flattening.npz` | Flat normalizado, medias, desviaciones, fecha, número de blancos y geometría. |
| `calibration_manifest.json` | Identidad, hashes SHA-256, fuentes y entorno de software. |

`calibration_id` es el SHA-256 de la descripción ordenada de los artefactos. Cambiar cualquier artefacto produce una identidad distinta o un fallo de verificación.

El manifiesto se escribe de forma atómica. Para cada fuente registra nombre, tamaño y SHA-256; para cada artefacto registra tamaño y SHA-256. También almacena fechas UTC, commit Git, Python y plataforma.

## CSV de resultados, esquema 3

Todas las filas contienen `schema_version = 3`. La procedencia se captura al medir y pertenece a cada resultado, no a la configuración seleccionada al exportar. La exportación valida el conjunto completo antes de reemplazar atómicamente el destino.

| Columna | Significado |
|---|---|
| `Filename` | Archivo de imagen, sin alterar la ruta fuente. |
| `Date` | Fecha extraída de metadatos o introducida por el usuario. |
| `Film` | Identificador de la película detectada. |
| `Circle` | Identificador de la ROI. |
| `doses_per_channel` | Dosis R, G y B o valor del canal disponible. |
| `STD_doses_per_channel` | Desviación espacial por canal. |
| `average` | Estimación combinada, con CTR aplicado si estaba activo. |
| `standard_uncertainty_average` | Incertidumbre estándar de `average`. |
| `expanded_uncertainty_k1.96` | $1.96\,u(\mathrm{average})$; cobertura normal aproximada. |
| `pixel_count` | Píxeles incluidos por la geometría de la ROI. |
| `uncertainty_calculation_method` | Método de combinación de canales. |
| `channel_weights` | Pesos normalizados R, G y B. |
| `calibration_id` | Identidad del conjunto de calibración. |
| `calibration_integrity` | Estado registrado al medir: `verified`, `legacy-unverified` o `unknown` si no consta procedencia. |
| `units` | `Gy` para dosis calibrada; `scanner_intensity` para intensidad sin calibrar. |
| `valid_pixel_counts` | Número de píxeles finitos por canal; puede diferir de `pixel_count`. |
| `measurement_status` | `valid`, `partial_pixels`, `partial_channels`, `uncertainty_unavailable` o `invalid`. |
| `x`, `y`, `radius` | Centro y radio de la ROI en píxeles de la imagen original. |
| `dose_correction_factor` | Factor multiplicativo aplicado a la medida. |
| `flat_applied` | Indica si se aplicó corrección espacial. |
| `ctr_context` | Contexto JSON del control utilizado, valor e incertidumbre. |

Las listas multicanal se serializan como valores separados por comas dentro del campo CSV. `expanded_uncertainty_k1.96` es una incertidumbre expandida; no sustituye una evaluación del factor de cobertura cuando la distribución o los grados de libertad requieren otro tratamiento.

## Configuración

`rc_config.json` se utiliza en la raíz del proyecto como configuración local. Las claves ausentes se completan con los valores predeterminados. La escritura reemplaza el archivo de forma atómica.

| Clave | Valor predeterminado | Efecto |
|---|---:|---|
| `uncertainty_estimation_method` | `dersimonian_laird` | Combinación RGB. |
| `calibration_folder` | `default` | Subcarpeta activa de calibración. |
| `allow_calibration_extrapolation` | `false` | Autoriza dosis fuera del dominio ajustado. |
| `calibration_extrapolation_margin_fraction` | `0.0` | Margen del intervalo cuando la extrapolación no está autorizada. |
| `allow_flat_field_resize` | `false` | Autoriza interpolar un flat con otra geometría. |
| `use_multithreading` | `true` | Activa trabajo paralelo compatible. |
| `num_threads` | CPU disponibles | Número de trabajadores. |
| `check_updates_on_startup` | `true` | Consulta actualizaciones al iniciar. |

La carga de imágenes se serializa para proteger el estado mutable del procesador. Las operaciones de Tk y la creación de `PhotoImage` se realizan en el hilo de interfaz.

## Precisión de imagen

- `read_image_unchanged` decodifica sin reducir profundidad y admite rutas Unicode en Windows.
- `storage_bit_depth` usa el tipo NumPy, no el máximo observado.
- `write_tiff_unchanged` conserva el tipo entero y el orden RGB de la API.
- La visualización puede escalarse, pero el análisis usa la matriz de resolución completa.
- Las imágenes integrales se calculan con precisión suficiente o se omiten cuando no son aplicables.

## Validación automatizada

La suite cubre:

- ida y vuelta del modelo racional y control de dominio;
- propagación de covarianza y combinación RGB;
- CTR independiente, correlacionado y autorresta;
- regresión ponderada y coordenadas de correlación de fase;
- manifiestos e identidad de calibración;
- exportación estrictamente numérica;
- TIFF Unicode `uint16`, profundidad de bits y geometría flat-field;
- configuración atómica, ciclo de vida de plugins y seguridad de rutas;
- actualizaciones Git con cambios locales.

La suite verifica consistencia interna del software. La validación experimental del sistema película–escáner debe realizarse con materiales y condiciones representativos del uso previsto.
