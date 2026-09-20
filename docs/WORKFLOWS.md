# Flujos de calibración y análisis

## Preparación de las imágenes

Use un protocolo de escaneo constante para blancos, películas de calibración y muestras:

- misma resolución, profundidad de bits y modo de color;
- misma orientación y región del cristal del escáner;
- condiciones de calentamiento y tiempo postirradiación definidos;
- archivos TIFF sin conversión con pérdida.

El lector conserva el `dtype` almacenado. Una imagen oscura `uint16` se trata como 16 bits aunque su máximo observado sea bajo. Los nombres y rutas Unicode no se modifican.

## Crear una calibración

Abra `Tools → Calibration Wizard` y seleccione una carpeta de calibración.

### 1. Uniformidad

1. Cargue uno o varios blancos obtenidos con el mismo protocolo.
2. Revise media, desviación, extremos y mapa espacial por canal.
3. Aplique la normalización para generar `field_flattening.npz`.

Cuando se promedian varios blancos, se escribe también `master_flat.tif` junto a sus fuentes. El archivo preserva 8 o 16 bits. Puede ejecutarse un flujo exclusivo de uniformidad para actualizar el flat-field.

### 2. Dosis–respuesta

1. Cargue los TIFF de calibración.
2. Confirme la dosis asociada a cada imagen, incluido el punto de dosis cero si existe.
3. Defina y mida las ROIs de cada película.
4. Revise los puntos R, G y B y ejecute el ajuste no lineal.
5. Guarde los resultados.

La carpeta contiene `calibration_data.csv`, `fit_parameters.csv` y, si se calculó, `field_flattening.npz`. `calibration_manifest.json` registra hashes, fuentes, intervalo de dosis, método, versión de Python, plataforma y commit.

Los parámetros introducidos manualmente no disponen de una covarianza estimada por el ajuste. En ese caso la conversión puede calcular dosis, pero la contribución paramétrica a la incertidumbre queda no disponible.

## Aplicar calibración

1. Seleccione `calibration_folder` en Settings.
2. Cargue la imagen.
3. Active flat-field y calibración según el análisis.

Antes de usar los artefactos, el programa verifica los hashes del manifiesto cuando existe. Una discrepancia bloquea el artefacto afectado. Una carpeta sin manifiesto se identifica como no verificada.

Cada canal genera una matriz de dosis, una máscara de validez y una máscara de extrapolación. Con la configuración predeterminada, los valores fuera del intervalo calibrado no participan en las ROIs.

## Medida manual

Seleccione una ROI circular, rectangular o de línea y sitúela sobre la imagen. El panel presenta por canal:

- media;
- desviación espacial;
- incertidumbre estándar;
- número de píxeles válidos;
- media combinada RGB e incertidumbre, si procede.

Los histogramas conservan como máximo 1000 muestras elegidas de forma determinista. Este muestreo afecta a la visualización, no a los estadísticos de la ROI.

## AutoMeasurements

1. Añada uno o varios archivos TIFF.
2. Ajuste, si es necesario, los umbrales de película y círculos.
3. Ejecute la detección.
4. Revise contornos y medidas antes de exportar.

Los círculos detectados se agrupan por filas usando su coordenada Y y se ordenan por X dentro de cada fila. El nombre `C{fila}{columna}` expresa esa posición; los círculos manuales usan el sufijo `M`. Una región sin medida numérica válida no se almacena como una medida cero.

Cada archivo mantiene su conjunto de películas, círculos, resultados y controles durante la navegación del lote.

## Controles CTR

Marque uno o varios círculos como CTR para corregir las medidas de su película. Un CTR global se aplica a todas las películas del conjunto. Los valores originales de precisión completa se conservan para que activar, desactivar o cambiar el control no acumule sustracciones.

El programa valida valor e incertidumbre antes de aplicar la corrección. Consulte las ecuaciones en [Modelo matemático e incertidumbre](MATHEMATICAL_MODEL.md#sustracción-de-controles-ctr).

## Analysis Tools

El complemento de análisis ofrece:

- comparación entre centro geométrico y centro de dosis;
- cuatro métodos de centroide;
- contornos de isodosis dentro de las ROIs;
- asociación de valores introducidos;
- regresión ponderada con intercepto libre o forzada al origen.

Revise visualmente los centroides y la geometría de cada ROI. La incertidumbre de la regresión sólo se informa cuando los datos identifican el modelo.

## Exportar

Use `Export CSV` al terminar la revisión. El exportador procesa los valores numéricos sin redondeo intermedio, rechaza campos científicos no numéricos e incorpora identificador y estado de integridad de la calibración. La presentación redondeada del TreeView no se usa como fuente si existe el valor numérico.

## Actualizaciones y plugins

Los plugins se pueden activar o desactivar desde la interfaz. Al desactivarlos se ejecuta su función de cierre para eliminar overlays y referencias.

La actualización integrada sólo acepta avance rápido de Git. Si el árbol contiene cambios, los guarda temporalmente incluyendo archivos no seguidos, actualiza y los restaura. Un conflicto conserva el guardado temporal para recuperación manual.
