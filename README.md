# QuPath Mask Exporter

Script de Groovy para exportar máscaras de anotaciones desde proyectos de QuPath, optimizado para imágenes de patología digital de gran tamaño (WSI - Whole Slide Images).

## El Problema: Imágenes Gigantes en Patología Digital

Las imágenes de microscopía de tejidos (WSI) son extremadamente grandes:

| Característica | Valor típico |
|----------------|--------------|
| Resolución | 50,000 - 150,000+ píxeles por lado |
| Píxeles totales | 2.5 - 22+ mil millones |
| Tamaño sin comprimir | 7 - 66+ GB (solo 1 canal uint8) |

Exportar máscaras de anotaciones de estas imágenes con métodos convencionales causa **errores de memoria** (`OutOfMemoryError: Java heap space`) porque intentan cargar la imagen completa en RAM antes de escribirla.

## La Solución: Escritura por Tiles

Este script utiliza `OMEPyramidWriter` de QuPath/Bio-Formats para escribir las máscaras **tile a tile** (por ejemplo, en bloques de 512x512 píxeles), sin necesidad de cargar toda la imagen en memoria.

```
┌─────────────────────────────────┐
│  Imagen Original (100k x 100k)  │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     LabeledImageServer          │
│  (genera etiquetas por tile)    │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     OMEPyramidWriter            │
│  (escribe tile a tile al TIFF)  │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│   TIFF Tileado + Comprimido     │
│      (eficiente en disco)       │
└─────────────────────────────────┘
```

> **Nota:** Este proceso es transparente para el usuario. El script se encarga de todo automáticamente.

## Qué Hace Este Script

1. **Recorre todas las imágenes** de un proyecto QuPath
2. **Genera una máscara por imagen** donde cada píxel tiene el valor de su clase
3. **Respeta la resolución original** de la imagen (sin downsampling)
4. **Exporta en formato uint8** (valores 0-255, suficiente para hasta 255 clases)
5. **Aplica optimizaciones** para archivos eficientes en disco y memoria

## Formato de Salida

| Propiedad | Valor |
|-----------|-------|
| Formato | OME-TIFF |
| Tipo de píxel | `uint8` |
| Canales | 1 (single-channel con índices de clase) |
| Estructura | Tileado (512x512 por defecto) |
| Compresión | LZW (lossless) |
| Resolución | Original (1:1) |

## Clases Definidas

El script asigna un valor entero (1-24) a cada clase de anotación. El fondo (sin anotación) tiene valor 0.

| ID | Clase |
|----|-------|
| 0 | Fondo (sin anotación) |
| 1 | Abnormal secretions |
| 2 | Adipose tissue |
| 3 | Artifact |
| 4 | Atypical intraductal proliferation |
| 5 | Bening gland |
| 6 | Blood vessels |
| 7 | Fibromuscular bundles |
| 8 | High grade prostatic intraepithelial neoplasia (HGPIN) |
| 9 | Immune cells |
| 10 | Intestinal glands and mucus |
| 11 | Intraductal carcinoma |
| 12 | Mitosis |
| 13 | Muscle |
| 14 | Necrosis |
| 15 | Negative |
| 16 | Nerve |
| 17 | Nerve ganglion |
| 18 | Normal secretions |
| 19 | Prominent nucleolus |
| 20 | Red blood cells |
| 21 | Seminal vesicle |
| 22 | Sin clasificación |
| 23 | Stroma |
| 24 | Tumor |

## Uso

### Requisitos

- QuPath 0.6.x o superior
- Proyecto con imágenes anotadas

### Ejecución

1. Abre tu proyecto en QuPath
2. Ve a **Automate > Script editor**
3. Copia el contenido de `export_masks.groovy`
4. Modifica la variable `OUTPUT_DIR` con tu ruta de destino
5. Ejecuta con **Run > Run** o `Ctrl+R`

### Configuración

```groovy
// Directorio de salida para las máscaras
def OUTPUT_DIR = "/ruta/a/tu/carpeta/de/salida"

// Tamaño de tile (512 es un buen balance)
def TILE_SIZE = 512
```

## Optimizaciones Aplicadas

### Eficiencia en Memoria

- **Escritura por tiles:** Nunca carga la imagen completa en RAM
- **Liberación de recursos:** Cierra servidores y ejecuta GC entre imágenes
- **Paralelización:** Usa múltiples threads para lectura/escritura

### Eficiencia en Disco

- **Compresión LZW:** Reduce significativamente el tamaño sin pérdida
- **Estructura tileada:** Permite lectura parcial sin cargar todo el archivo
- **Single-channel uint8:** Mínimo espacio por píxel (1 byte)

### Compatibilidad

- **OME-TIFF:** Estándar abierto compatible con:
  - ImageJ / FIJI
  - Python (tifffile, openslide, etc.)
  - MATLAB
  - Cualquier lector Bio-Formats

## Visor Incluido (view_mask.py)

Este repositorio incluye un visor interactivo en Python para explorar las máscaras sin saturar la RAM.

### Instalación

```bash
pip install -r requirements.txt
```

### Uso

```bash
# Vista general (con downsampling automático)
python view_mask.py /ruta/a/mascara.tif

# Ver región específica (x, y, ancho, alto)
python view_mask.py mascara.tif --region 10000 10000 2048 2048

# Controlar el downsampling
python view_mask.py mascara.tif --downsample 8
```

El visor muestra automáticamente una **leyenda con las clases presentes** en la región visualizada.

## Tips para Abrir Máscaras Grandes

Las máscaras exportadas son **OME-TIFF tileados**. Esto significa:

1. **No cargar todo en memoria:** Nunca hagas `mask = tifffile.imread(path)` en imágenes grandes
2. **Usar acceso por tiles:** `tifffile` lee solo los tiles necesarios automáticamente
3. **Cargar regiones:** Usa slicing para cargar solo lo que necesitas

```python
import tifffile

# INCORRECTO para imágenes grandes (carga todo en RAM)
# mask = tifffile.imread("mascara.tif")

# INCORRECTO: asarray() también carga todo antes del slice
# region = tif.pages[0].asarray()[1000:2000, 1000:2000]

# CORRECTO: Leer tiles individuales para regiones específicas
with tifffile.TiffFile("mascara.tif") as tif:
    page = tif.pages[0]
    
    # Ver dimensiones sin cargar
    print(f"Tamaño: {page.shape}")
    print(f"Tileado: {page.is_tiled}")
    if page.is_tiled:
        print(f"Tile size: {page.tilewidth}x{page.tilelength}")
    
    # Para regiones pequeñas en TIFFs tileados, usar el visor incluido
    # que lee solo los tiles necesarios
```

El script `view_mask.py` incluido lee tiles individuales sin cargar toda la imagen.

### ImageJ / FIJI

```
File > Open > [seleccionar el TIFF]
```

El archivo se abrirá como imagen de 8-bit donde cada valor de gris corresponde a una clase.

Para imágenes muy grandes, usa **Bio-Formats Importer**:
```
Plugins > Bio-Formats > Bio-Formats Importer
```
Y selecciona "Use virtual stack" para no cargar todo en memoria.

## Troubleshooting

### Error: OutOfMemoryError

Si aún aparece este error:

1. Aumenta la memoria de QuPath: **Edit > Preferences > Memory**
2. Reduce `TILE_SIZE` a 256
3. Cierra otras aplicaciones que consuman RAM

### Error: No hay proyecto abierto

Asegúrate de tener un proyecto QuPath abierto (no solo una imagen individual).

## Licencia

MIT License
