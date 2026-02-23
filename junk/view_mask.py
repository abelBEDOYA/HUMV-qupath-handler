#!/usr/bin/env python3
"""
QuPath Mask Viewer

Visor interactivo para máscaras de anotaciones exportadas desde QuPath.
Permite navegar por imágenes muy grandes sin saturar la RAM,
cargando solo la región visible usando acceso por tiles.

Uso:
    python view_mask.py /ruta/a/mascara.tif

Controles:
    - Flechas / Arrastrar: Mover por la imagen
    - Scroll / +/-: Zoom
    - Home: Volver a vista completa
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import tifffile

# =====================================================
# DEFINICIÓN DE CLASES (debe coincidir con el script de QuPath)
# =====================================================

CLASS_NAMES = {
    0: "Background",
    1: "Abnormal secretions",
    2: "Adipose tissue",
    3: "Artifact",
    4: "Atypical intraductal proliferation",
    5: "Bening gland",
    6: "Blood vessels",
    7: "Fibromuscular bundles",
    8: "HGPIN",
    9: "Immune cells",
    10: "Intestinal glands and mucus",
    11: "Intraductal carcinoma",
    12: "Mitosis",
    13: "Muscle",
    14: "Necrosis",
    15: "Negative",
    16: "Nerve",
    17: "Nerve ganglion",
    18: "Normal secretions",
    19: "Prominent nucleolus",
    20: "Red blood cells",
    21: "Seminal vesicle",
    22: "Sin clasificación",
    23: "Stroma",
    24: "Tumor",
}

# Colores para cada clase (puedes personalizar)
CLASS_COLORS = [
    "#000000",  # 0: Background (negro)
    "#FF6B6B",  # 1: Abnormal secretions
    "#FFE66D",  # 2: Adipose tissue
    "#4ECDC4",  # 3: Artifact
    "#95E1D3",  # 4: Atypical intraductal proliferation
    "#F38181",  # 5: Bening gland
    "#AA96DA",  # 6: Blood vessels
    "#FCBAD3",  # 7: Fibromuscular bundles
    "#A8D8EA",  # 8: HGPIN
    "#FF9F43",  # 9: Immune cells
    "#6A0572",  # 10: Intestinal glands and mucus
    "#AB83A1",  # 11: Intraductal carcinoma
    "#E84545",  # 12: Mitosis
    "#903749",  # 13: Muscle
    "#53354A",  # 14: Necrosis
    "#2B2E4A",  # 15: Negative
    "#E9D5CA",  # 16: Nerve
    "#BBDED6",  # 17: Nerve ganglion
    "#61C0BF",  # 18: Normal secretions
    "#FAE3D9",  # 19: Prominent nucleolus
    "#CC0000",  # 20: Red blood cells
    "#FFB6B9",  # 21: Seminal vesicle
    "#8AC6D1",  # 22: Sin clasificación
    "#F9ED69",  # 23: Stroma
    "#B83B5E",  # 24: Tumor
]


class MaskViewer:
    """
    Visor de máscaras que carga solo regiones visibles usando tiles.
    
    Usa tifffile para leer tiles individuales de TIFFs tileados,
    sin cargar toda la imagen en memoria.
    """
    
    def __init__(self, mask_path: str, max_viewport_size: int = 4096):
        """
        Args:
            mask_path: Ruta al archivo TIFF de la máscara
            max_viewport_size: Tamaño máximo de región a cargar (ancho o alto)
        """
        self.mask_path = mask_path
        self.max_viewport_size = max_viewport_size
        
        # Abrir archivo (solo metadatos, no carga píxeles)
        self.tif = tifffile.TiffFile(mask_path)
        
        # Obtener la primera página/serie
        self.page = self.tif.pages[0]
        
        # Dimensiones totales
        self.height = self.page.shape[0]
        self.width = self.page.shape[1] if len(self.page.shape) > 1 else 1
        
        # Info de tiles
        self.is_tiled = self.page.is_tiled
        if self.is_tiled:
            self.tile_width = self.page.tilewidth
            self.tile_height = self.page.tilelength
        else:
            self.tile_width = self.width
            self.tile_height = self.page.rowsperstrip or self.height
        
        print(f"Máscara: {mask_path}")
        print(f"Dimensiones: {self.width} x {self.height}")
        print(f"Tipo: {self.page.dtype}")
        print(f"Tileado: {self.is_tiled}")
        if self.is_tiled:
            print(f"Tamaño de tile: {self.tile_width} x {self.tile_height}")
        
        # Estado de visualización
        self.fig = None
        self.ax = None
        self.im = None
        
    def _load_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Carga una región específica de la máscara leyendo solo los tiles necesarios.
        """
        # Asegurar límites
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        width = min(width, self.width - x)
        height = min(height, self.height - y)
        
        if not self.is_tiled:
            # Si no es tileado, tenemos que cargar todo (no hay otra opción)
            print("AVISO: TIFF no tileado, cargando imagen completa...")
            full_data = self.page.asarray()
            return self._ensure_2d(full_data)[y:y+height, x:x+width]
        
        # Calcular qué tiles necesitamos
        tile_x_start = x // self.tile_width
        tile_y_start = y // self.tile_height
        tile_x_end = (x + width - 1) // self.tile_width + 1
        tile_y_end = (y + height - 1) // self.tile_height + 1
        
        n_tiles_x = tile_x_end - tile_x_start
        n_tiles_y = tile_y_end - tile_y_start
        
        # Crear array para el resultado
        result_height = n_tiles_y * self.tile_height
        result_width = n_tiles_x * self.tile_width
        
        result = np.zeros((result_height, result_width), dtype=self.page.dtype)
        
        tiles_per_row = (self.width + self.tile_width - 1) // self.tile_width
        
        for iy, ty in enumerate(range(tile_y_start, tile_y_end)):
            for ix, tx in enumerate(range(tile_x_start, tile_x_end)):
                # Posición en el resultado
                ry = iy * self.tile_height
                rx = ix * self.tile_width
                
                # Leer el tile
                try:
                    tile_data = self._read_tile(tx, ty, tiles_per_row)
                    tile_2d = self._ensure_2d(tile_data)
                    th, tw = tile_2d.shape
                    
                    # Asegurar que no excedemos los límites
                    th = min(th, result_height - ry)
                    tw = min(tw, result_width - rx)
                    
                    result[ry:ry+th, rx:rx+tw] = tile_2d[:th, :tw]
                except Exception as e:
                    print(f"Error leyendo tile ({tx}, {ty}): {e}")
        
        # Recortar al área solicitada
        offset_x = x - tile_x_start * self.tile_width
        offset_y = y - tile_y_start * self.tile_height
        
        return result[offset_y:offset_y+height, offset_x:offset_x+width]
    
    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        """
        Asegura que el array sea 2D, eliminando dimensiones extra.
        """
        # Squeeze elimina dimensiones de tamaño 1
        arr = np.squeeze(arr)
        
        # Si sigue teniendo más de 2 dimensiones, tomar el primer canal
        while arr.ndim > 2:
            arr = arr[0]
        
        return arr
    
    def _read_tile(self, tx: int, ty: int, tiles_per_row: int) -> np.ndarray:
        """
        Lee un tile específico del TIFF.
        """
        tile_index = ty * tiles_per_row + tx
        
        # Verificar que el índice es válido
        if tile_index >= len(self.page.dataoffsets):
            return np.zeros((self.tile_height, self.tile_width), dtype=self.page.dtype)
        
        offset = self.page.dataoffsets[tile_index]
        bytecount = self.page.databytecounts[tile_index]
        
        # Si el tile está vacío, devolver ceros
        if bytecount == 0:
            return np.zeros((self.tile_height, self.tile_width), dtype=self.page.dtype)
        
        # Leer y decodificar el tile
        fh = self.tif.filehandle
        fh.seek(offset)
        data = fh.read(bytecount)
        
        # Decodificar según la compresión
        tile = self.page.decode(data, tile_index)[0]
        
        return tile
    
    def _load_downsampled(self, downsample: int = 1) -> np.ndarray:
        """
        Carga la imagen con downsampling para vista general.
        
        Lee tiles espaciados y los combina.
        """
        step = max(1, downsample)
        
        if not self.is_tiled or step == 1:
            # Para downsampling pequeño o no tileado, cargar y submuestrear
            # Pero limitamos a una región manejable
            if self.width * self.height > 100_000_000:  # >100 megapixels
                # Muy grande, leer solo algunos tiles
                return self._load_sparse_overview(step)
            else:
                data = self._ensure_2d(self.page.asarray())
                return data[::step, ::step]
        
        return self._load_sparse_overview(step)
    
    def _load_sparse_overview(self, step: int) -> np.ndarray:
        """
        Crea una vista general leyendo tiles espaciados.
        """
        # Calcular dimensiones de salida
        out_h = (self.height + step - 1) // step
        out_w = (self.width + step - 1) // step
        
        result = np.zeros((out_h, out_w), dtype=self.page.dtype)
        
        if not self.is_tiled:
            # Sin tiles, necesitamos cargar todo
            print("AVISO: Cargando imagen completa para downsampling...")
            data = self._ensure_2d(self.page.asarray())
            return data[::step, ::step]
        
        # Leer tiles espaciados
        tiles_per_row = (self.width + self.tile_width - 1) // self.tile_width
        tiles_per_col = (self.height + self.tile_height - 1) // self.tile_height
        
        # Determinar cada cuántos tiles leer
        tile_step = max(1, step // min(self.tile_width, self.tile_height))
        if tile_step < 1:
            tile_step = 1
        
        print(f"Leyendo tiles (cada {tile_step} tiles)...")
        
        for ty in range(0, tiles_per_col, tile_step):
            for tx in range(0, tiles_per_row, tile_step):
                try:
                    tile = self._read_tile(tx, ty, tiles_per_row)
                    tile_2d = self._ensure_2d(tile)
                    
                    # Posición en el original
                    orig_y = ty * self.tile_height
                    orig_x = tx * self.tile_width
                    
                    # Submuestrear el tile
                    tile_ds = tile_2d[::step, ::step]
                    
                    # Posición en el resultado
                    out_y = orig_y // step
                    out_x = orig_x // step
                    
                    # Copiar
                    th, tw = tile_ds.shape
                    if out_y + th <= out_h and out_x + tw <= out_w:
                        result[out_y:out_y+th, out_x:out_x+tw] = tile_ds
                except Exception:
                    pass
        
        return result
    
    def view(self, region: tuple = None, downsample: int = None):
        """
        Visualiza la máscara o una región de ella.
        
        Args:
            region: Tupla (x, y, width, height) de la región a mostrar.
                   Si es None, muestra vista general con downsampling.
            downsample: Factor de downsampling para vista general.
                       Se calcula automáticamente si es None.
        """
        # Crear colormap discreto
        cmap = ListedColormap(CLASS_COLORS[:25])
        bounds = np.arange(-0.5, 25.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        
        if region is None:
            # Vista general: calcular downsampling automático
            if downsample is None:
                downsample = max(1, max(self.width, self.height) // self.max_viewport_size)
            
            print(f"Cargando vista general (downsample: {downsample}x)...")
            data = self._load_downsampled(downsample)
            title = f"Vista general ({self.width}x{self.height}, {downsample}x downsample)"
        else:
            # Región específica
            x, y, w, h = region
            print(f"Cargando región ({x}, {y}, {w}, {h})...")
            data = self._load_region(x, y, w, h)
            title = f"Región: x={x}, y={y}, w={w}, h={h}"
        
        print(f"Datos cargados: {data.shape}, {data.dtype}")
        print(f"Memoria usada: {data.nbytes / 1024 / 1024:.2f} MB")
        
        # Encontrar clases presentes
        unique_classes = np.unique(data)
        print(f"Clases en esta vista: {list(unique_classes)}")
        
        # Crear figura
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
        # Mostrar máscara
        self.im = self.ax.imshow(
            data, 
            cmap=cmap, 
            norm=norm,
            interpolation='nearest'
        )
        
        self.ax.set_title(title)
        self.ax.set_xlabel("X (píxeles)")
        self.ax.set_ylabel("Y (píxeles)")
        
        # Crear leyenda solo con clases presentes
        legend_patches = []
        for class_id in sorted(unique_classes):
            if class_id in CLASS_NAMES:
                patch = Patch(
                    facecolor=CLASS_COLORS[class_id],
                    edgecolor='black',
                    label=f"{class_id}: {CLASS_NAMES[class_id]}"
                )
                legend_patches.append(patch)
        
        # Posicionar leyenda fuera del gráfico
        self.ax.legend(
            handles=legend_patches,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            framealpha=0.9
        )
        
        plt.tight_layout()
        plt.show()
    
    def view_interactive(self):
        """
        Abre vista interactiva con zoom y pan usando matplotlib.
        """
        # Calcular downsampling inicial
        downsample = max(1, max(self.width, self.height) // self.max_viewport_size)
        
        print(f"Cargando para vista interactiva (downsample: {downsample}x)...")
        print("Usa el zoom de matplotlib para explorar regiones.")
        
        data = self._load_downsampled(downsample)
        
        print(f"Datos cargados: {data.shape}")
        print(f"Memoria usada: {data.nbytes / 1024 / 1024:.2f} MB")
        
        # Crear colormap
        cmap = ListedColormap(CLASS_COLORS[:25])
        bounds = np.arange(-0.5, 25.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Encontrar clases presentes
        unique_classes = np.unique(data)
        
        # Crear figura con toolbar de navegación
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
        self.im = self.ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            extent=[0, self.width, self.height, 0]  # Coordenadas reales
        )
        
        self.ax.set_title(f"Máscara: {self.width}x{self.height} (mostrando {downsample}x downsample)")
        self.ax.set_xlabel("X (píxeles)")
        self.ax.set_ylabel("Y (píxeles)")
        
        # Leyenda
        legend_patches = [
            Patch(facecolor=CLASS_COLORS[c], edgecolor='black', 
                  label=f"{c}: {CLASS_NAMES[c]}")
            for c in sorted(unique_classes) if c in CLASS_NAMES
        ]
        
        self.ax.legend(
            handles=legend_patches,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8
        )
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Cierra el archivo TIFF."""
        self.tif.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visor de máscaras de QuPath",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Vista general de la máscara
  python view_mask.py mascara.tif
  
  # Ver región específica (x, y, ancho, alto)
  python view_mask.py mascara.tif --region 10000 10000 2048 2048
  
Tips para imágenes grandes:
  - Las máscaras exportadas son OME-TIFF tileados
  - Este script lee solo los tiles necesarios
  - Usa --region para explorar áreas específicas sin cargar todo
  - El downsampling automático permite ver la imagen completa
        """
    )
    
    parser.add_argument("mask_path", help="Ruta al archivo TIFF de la máscara")
    parser.add_argument(
        "--region", "-r",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Región a visualizar (x, y, ancho, alto)"
    )
    parser.add_argument(
        "--downsample", "-d",
        type=int,
        default=None,
        help="Factor de downsampling para vista general"
    )
    parser.add_argument(
        "--max-size", "-m",
        type=int,
        default=4096,
        help="Tamaño máximo de viewport (default: 4096)"
    )
    
    args = parser.parse_args()
    
    # Crear visor
    viewer = MaskViewer(args.mask_path, max_viewport_size=args.max_size)
    
    try:
        if args.region:
            # Ver región específica
            viewer.view(region=tuple(args.region))
        else:
            # Vista interactiva
            viewer.view_interactive()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
