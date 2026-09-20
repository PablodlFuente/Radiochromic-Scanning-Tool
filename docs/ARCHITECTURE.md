# Arquitectura y verificación

## Responsabilidades

| Componente | Responsabilidad |
|---|---|
| `main.py`, `app/rc_analyzer.py` | Arranque, registro y ciclo de vida de la aplicación Tk. |
| `app/ui/` | Ventanas, eventos, presentación y coordinación de acciones del usuario. |
| `app/core/image_processor.py` | Estado de imagen, aplicación de correcciones, ROI y representación. |
| `app/core/dosimetry.py` | Modelo racional, dominio, propagación paramétrica y combinación RGB. |
| `app/core/calibration_manifest.py` | Identidad de artefactos, procedencia y verificación. |
| `app/calibration/` | Asistente, extracción de puntos y ajuste dosis–respuesta. |
| `app/utils/` | Lectura y escritura, configuración, archivos y actualización Git. |
| `app/plugins/` | Descubrimiento, activación, notificaciones y cierre de complementos. |
| `custom_plugins/auto_measurements/` | Detección, resultados por archivo, CTR y exportación. |
| `custom_plugins/analysis_tools/` | Centroides, isodosis y regresión. |
| `tests/` | Pruebas numéricas, regresiones e integración. |

El flujo de análisis es imagen original → flat-field opcional → dosis opcional → ROI → combinación de canales → CTR opcional → exportación. El binning pertenece únicamente a la representación.

## Estado y concurrencia

El procesador protege las operaciones sobre su estado con un bloqueo reentrante. Las solicitudes de carga llevan identificador y las respuestas obsoletas se descartan. El panel entrega resultados de trabajadores a Tk mediante una cola y comprueba la revisión del estado antes de mostrarlos. Al cargar otra imagen, los complementos invalidan sus resultados visibles antes de restaurar, cuando corresponda, la copia almacenada por archivo.

Las medidas guardan su procedencia y los conjuntos de resultados por archivo se copian de forma profunda. La exportación no recalcula controles ni sustituye la procedencia histórica por la configuración activa. Cada procesador mantiene su propio directorio temporal.

## Estrategia de pruebas

Ejecute desde la raíz:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app custom_plugins tests main.py
python -m pip check
```

Las pruebas numéricas utilizan ejemplos analíticos, inversiones conocidas y transformaciones de unidades. Las pruebas de integración verifican archivos reales temporales, TIFF de 16 bits, manifiestos y repositorios Git aislados. La prueba Tk utiliza un bucle de eventos real, carga dos imágenes y verifica la invalidación del estado; se omite explícitamente si no existe una pantalla Tk disponible.

Los dobles de prueba permiten provocar fallos de procesamiento y probar la conservación del destino al exportar. No sustituyen la evaluación de interacción visual, estrés de concurrencia, memoria con imágenes grandes, GPU ni validación con películas y dosis de referencia independientes.

La interfaz y algunos controladores de complementos concentran varias responsabilidades. Para ampliar funcionalidad conviene mantener el cálculo en funciones independientes de Tk y los resultados como datos explícitos, con pruebas en ese límite. La validación científica del sistema completo requiere el protocolo y las restricciones descritos en el [modelo matemático](MATHEMATICAL_MODEL.md).
