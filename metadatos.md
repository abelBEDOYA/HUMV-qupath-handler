# Metadatos del Dataset

## Descripción

Dataset de imágenes histopatológicas de próstata con máscaras de segmentación semántica multiclase.

**Origen:** Hospital Universitario Marqués de Valdecilla (HUMV), Santander, España.

**Etiquetado:** Realizado por patólogo especializado utilizando QuPath.

**Número de muestras:** 37 imágenes con sus correspondientes 37 máscaras de segmentación.

---

## Estructura del Dataset

```
dataset/
├── images/          
├── masks/           
├── clases.txt
└── metadatos.md
```

---

## Imágenes (`images/`)

- **Formato:** OME-TIFF piramidal
- **Extensión:** `.ome.tif`
- **Tipo de píxel:** RGB (3 canales, 8 bits por canal)
- **Compresión:** LZW (lossless)
- **Estructura piramidal:** Múltiples niveles de resolución
  - Nivel 0: Resolución completa (downsample 1×)
  - Nivel 1: Downsample 4×
  - Nivel 2: Downsample 16×
  - Nivel 3: Downsample 64×
  - ... (hasta que la dimensión mayor sea < 1024 px)

Las imágenes piramidales permiten visualización y procesamiento eficiente a diferentes escalas sin cargar la imagen completa en memoria.

---

## Máscaras (`masks/`)

- **Formato:** OME-TIFF piramidal
- **Extensión:** `__mask_multiclass.ome.tif`
- **Tipo de píxel:** uint8 (1 canal, 8 bits)
- **Compresión:** LZW (lossless)
- **Estructura piramidal:** Mismos niveles que las imágenes correspondientes

Cada píxel contiene un valor entero (0-24) que indica la clase a la que pertenece.

**Correspondencia imagen-máscara:**
- `images/IMAGEN_001.ome.tif` → `masks/IMAGEN_001__mask_multiclass.ome.tif`

---

## Clases de Segmentación

| ID | Clase |
|----|-------|
| 0 | Background (sin anotación) |
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

**Total:** 25 clases (incluyendo background)

---

## Notas Técnicas

### Lectura de las imágenes

Las imágenes OME-TIFF piramidales pueden leerse con bibliotecas como:
- **Python:** `tifffile`, `openslide-python`, `pyvips`
- **Java:** Bio-Formats, QuPath

**Ejemplo con tifffile (Python):**

```python
import tifffile

# Abrir sin cargar en memoria
with tifffile.TiffFile("imagen.ome.tif") as tif:
    # Ver niveles disponibles
    for i, level in enumerate(tif.series[0].levels):
        print(f"Nivel {i}: {level.shape}")
    
    # Leer nivel específico (ej: nivel 2 = downsample 16×)
    data = tif.series[0].levels[2].asarray()
```

### Alineación imagen-máscara

Las máscaras tienen exactamente los mismos niveles piramidales que sus imágenes correspondientes, garantizando alineación 1:1 a cualquier resolución.

### Solapamientos

En zonas donde múltiples anotaciones se solapan, la máscara contiene el valor de la clase con mayor prioridad (las clases con ID más alto tienen mayor prioridad).

---

## Licencia y Uso

[Especificar términos de uso y licencia]

---

## Contacto

[Información de contacto]

---

## Cita

Si utiliza este dataset, por favor cite:

[Referencia bibliográfica]
