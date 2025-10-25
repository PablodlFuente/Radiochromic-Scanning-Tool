# Backup de AutoMeasurements Plugin

Esta carpeta contiene copias de seguridad del plugin AutoMeasurements antes de la migración a estructura de paquete multi-archivo.

## Archivos

### auto_measurements_original.py (201,271 bytes)
- Versión original del plugin en archivo único
- Fecha: 24 de octubre de 2025
- 4,542 líneas de código
- **Esta es la versión que funcionaba antes de la migración**

### auto_measurements_single_file_backup.py (201,271 bytes)
- Backup adicional creado durante el proceso de migración
- Contenido idéntico a `auto_measurements_original.py`

## Restauración

Si necesitas volver a la versión de archivo único:

1. Detén la aplicación
2. Elimina la carpeta `custom_plugins/auto_measurements/` (el paquete actual)
3. Copia `auto_measurements_original.py` a `custom_plugins/`
4. Renómbralo a `auto_measurements.py`
5. Reinicia la aplicación

## Nueva Versión (Paquete Multi-Archivo)

La versión actual está en: `custom_plugins/auto_measurements/` (carpeta)

Estructura:
```
auto_measurements/
├── __init__.py           - Plugin interface
├── models/               - Data classes y constants
├── core/                 - 6 clases principales  
└── ui/                   - AutoMeasurementsTab (interfaz)
```

**Ventajas:**
- Mejor organización del código
- Más fácil de mantener y extender
- Separación clara de responsabilidades
- Mismo rendimiento y funcionalidad

---
*Fecha de migración: 24 de octubre de 2025*
