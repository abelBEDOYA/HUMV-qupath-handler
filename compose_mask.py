#!/usr/bin/env python3
"""
QuPath Mask Composer

Genera una imagen compuesta de toda la máscara redimensionada a un tamaño
manejable, leyendo por tiles para no saturar la RAM.

Uso:
    python compose_mask.py mascara.tif -o output.png
    python compose_mask.py mascara.tif -o output.png --width 10000
    python compose_mask.py mascara.tif --width -1  # Ver imagen completa (cuidado con RAM)

Opciones:
    --width, -w: Ancho de la imagen de salida (default: 10000, -1 para tamaño original)
    --output, -o: Archivo de salida (PNG o TIFF)
    --show: Mostrar la imagen después de generarla
    --no-legend: No añadir leyenda
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import tifffile
from pathlib import Path


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

CLASS_COLORS_HEX = [
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


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convierte color hex a RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


CLASS_COLORS_RGB = [hex_to_rgb(c) for c in CLASS_COLORS_HEX]


class MaskComposer:
    """Compone una máscara grande en una imagen redimensionada."""
    
    def __init__(self, mask_path: str):
        self.mask_path = mask_path
        self.tif = tifffile.TiffFile(mask_path)
        self.page = self.tif.pages[0]
        
        self.height = self.page.shape[0]
        self.width = self.page.shape[1] if len(self.page.shape) > 1 else 1
        
        self.is_tiled = self.page.is_tiled
        if self.is_tiled:
            self.tile_width = self.page.tilewidth
            self.tile_height = self.page.tilelength
        else:
            self.tile_width = self.width
            self.tile_height = self.page.rowsperstrip or self.height
        
        print(f"Máscara: {mask_path}")
        print(f"Dimensiones originales: {self.width} x {self.height}")
        print(f"Tileado: {self.is_tiled}")
        if self.is_tiled:
            print(f"Tamaño de tile: {self.tile_width} x {self.tile_height}")
    
    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = arr[0]
        return arr
    
    def _read_tile(self, tx: int, ty: int, tiles_per_row: int) -> np.ndarray:
        tile_index = ty * tiles_per_row + tx
        
        if tile_index >= len(self.page.dataoffsets):
            return np.zeros((self.tile_height, self.tile_width), dtype=self.page.dtype)
        
        offset = self.page.dataoffsets[tile_index]
        bytecount = self.page.databytecounts[tile_index]
        
        if bytecount == 0:
            return np.zeros((self.tile_height, self.tile_width), dtype=self.page.dtype)
        
        fh = self.tif.filehandle
        fh.seek(offset)
        data = fh.read(bytecount)
        
        tile = self.page.decode(data, tile_index)[0]
        return tile
    
    def compose(
        self, 
        target_width: int = 10000,
        output_path: str | None = None,
        show: bool = False,
        add_legend: bool = True
    ) -> np.ndarray | None:
        """
        Compone la máscara completa en una imagen redimensionada.
        
        Args:
            target_width: Ancho objetivo (-1 para tamaño original)
            output_path: Ruta de salida (PNG o TIFF)
            show: Mostrar la imagen con matplotlib
            add_legend: Añadir leyenda de clases
        
        Returns:
            Array RGB de la imagen compuesta
        """
        if target_width == -1:
            target_width = self.width
            target_height = self.height
            downsample = 1
            print(f"\nGenerando imagen a tamaño completo: {target_width} x {target_height}")
            print("AVISO: Esto puede usar mucha RAM.")
        else:
            aspect_ratio = self.height / self.width
            target_height = int(target_width * aspect_ratio)
            downsample = self.width / target_width
            print(f"\nGenerando imagen: {target_width} x {target_height} (downsample: {downsample:.2f}x)")
        
        output_rgb = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        classes_found: set[int] = set()
        
        if not self.is_tiled:
            print("Cargando imagen completa (no tileada)...")
            full_data = self._ensure_2d(self.page.asarray())
            classes_found = set(np.unique(full_data))
            
            if downsample != 1:
                from scipy.ndimage import zoom
                scale = 1.0 / downsample
                full_data = zoom(full_data, scale, order=0)
            
            for class_id in range(25):
                mask = full_data == class_id
                if np.any(mask):
                    r, g, b = CLASS_COLORS_RGB[class_id]
                    output_rgb[mask] = [r, g, b]
        else:
            tiles_per_row = (self.width + self.tile_width - 1) // self.tile_width
            tiles_per_col = (self.height + self.tile_height - 1) // self.tile_height
            total_tiles = tiles_per_row * tiles_per_col
            
            print(f"Procesando {total_tiles} tiles...")
            
            for ty in range(tiles_per_col):
                for tx in range(tiles_per_row):
                    try:
                        tile = self._read_tile(tx, ty, tiles_per_row)
                        tile_2d = self._ensure_2d(tile)
                        
                        classes_found.update(np.unique(tile_2d))
                        
                        orig_x = tx * self.tile_width
                        orig_y = ty * self.tile_height
                        
                        out_x_start = int(orig_x / downsample)
                        out_y_start = int(orig_y / downsample)
                        out_x_end = int(min((orig_x + self.tile_width), self.width) / downsample)
                        out_y_end = int(min((orig_y + self.tile_height), self.height) / downsample)
                        
                        out_w = out_x_end - out_x_start
                        out_h = out_y_end - out_y_start
                        
                        if out_w <= 0 or out_h <= 0:
                            continue
                        
                        if downsample > 1:
                            step = max(1, int(downsample))
                            tile_ds = tile_2d[::step, ::step]
                            tile_ds = tile_ds[:out_h, :out_w]
                        else:
                            tile_ds = tile_2d[:out_h, :out_w]
                        
                        actual_h, actual_w = tile_ds.shape
                        out_y_end = out_y_start + actual_h
                        out_x_end = out_x_start + actual_w
                        
                        if out_y_end > target_height:
                            actual_h = target_height - out_y_start
                            tile_ds = tile_ds[:actual_h, :]
                            out_y_end = target_height
                        if out_x_end > target_width:
                            actual_w = target_width - out_x_start
                            tile_ds = tile_ds[:, :actual_w]
                            out_x_end = target_width
                        
                        for class_id in np.unique(tile_ds):
                            mask = tile_ds == class_id
                            r, g, b = CLASS_COLORS_RGB[class_id]
                            
                            region = output_rgb[out_y_start:out_y_end, out_x_start:out_x_end]
                            region[mask] = [r, g, b]
                            
                    except Exception as e:
                        print(f"\nError en tile ({tx}, {ty}): {e}")
                        continue
                
                progress = (ty + 1) / tiles_per_col * 100
                print(f"\rProgreso: {progress:.1f}%", end="", flush=True)
            
            print()
        
        print(f"\nClases encontradas: {sorted(classes_found - {0})}")
        print(f"Tamaño imagen: {output_rgb.shape}")
        print(f"Memoria usada: {output_rgb.nbytes / 1024 / 1024:.2f} MB")
        
        if output_path:
            self._save_image(output_rgb, output_path, classes_found, add_legend)
        
        if show:
            self._show_image(output_rgb, classes_found, add_legend)
        
        return output_rgb
    
    def _save_image(
        self, 
        rgb_data: np.ndarray, 
        output_path: str,
        classes_found: set[int],
        add_legend: bool
    ) -> None:
        """Guarda la imagen con o sin leyenda."""
        output_path = Path(output_path)
        
        if add_legend:
            fig = self._create_figure_with_legend(rgb_data, classes_found)
            
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            print(f"Imagen con leyenda guardada en: {output_path}")
        else:
            img = Image.fromarray(rgb_data)
            img.save(output_path)
            print(f"Imagen guardada en: {output_path}")
    
    def _create_figure_with_legend(
        self, 
        rgb_data: np.ndarray, 
        classes_found: set[int]
    ) -> plt.Figure:
        """Crea figura con leyenda."""
        h, w = rgb_data.shape[:2]
        aspect = h / w
        fig_width = 16
        fig_height = fig_width * aspect
        
        fig, ax = plt.subplots(figsize=(fig_width, max(fig_height, 8)))
        
        ax.imshow(rgb_data)
        ax.set_title(f"Máscara: {self.width} x {self.height} → {w} x {h}")
        ax.axis('off')
        
        legend_patches = [
            Patch(
                facecolor=CLASS_COLORS_HEX[class_id],
                edgecolor='black',
                label=f"{class_id}: {CLASS_NAMES[class_id]}"
            )
            for class_id in sorted(classes_found) 
            if class_id in CLASS_NAMES and class_id != 0
        ]
        
        if legend_patches:
            ax.legend(
                handles=legend_patches,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
                framealpha=0.95
            )
        
        plt.tight_layout()
        return fig
    
    def _show_image(
        self, 
        rgb_data: np.ndarray, 
        classes_found: set[int],
        add_legend: bool
    ) -> None:
        """Muestra la imagen con matplotlib."""
        if add_legend:
            fig = self._create_figure_with_legend(rgb_data, classes_found)
        else:
            h, w = rgb_data.shape[:2]
            aspect = h / w
            fig_width = 14
            fig, ax = plt.subplots(figsize=(fig_width, fig_width * aspect))
            ax.imshow(rgb_data)
            ax.axis('off')
            plt.tight_layout()
        
        plt.show()
    
    def close(self):
        self.tif.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compone una máscara de QuPath en una imagen redimensionada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Generar imagen de 10k píxeles de ancho
  python compose_mask.py mascara.tif -o output.png
  
  # Generar imagen de 5000 píxeles de ancho
  python compose_mask.py mascara.tif -o output.png --width 5000
  
  # Ver imagen sin guardar
  python compose_mask.py mascara.tif --show
  
  # Imagen a tamaño completo (cuidado con RAM)
  python compose_mask.py mascara.tif -o full.png --width -1
  
  # Sin leyenda
  python compose_mask.py mascara.tif -o output.png --no-legend
        """
    )
    
    parser.add_argument("mask_path", help="Ruta al archivo TIFF de la máscara")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Archivo de salida (PNG, TIFF, JPG)"
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=10000,
        help="Ancho de la imagen de salida (default: 10000, -1 para original)"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Mostrar la imagen después de generarla"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="No añadir leyenda de clases"
    )
    
    args = parser.parse_args()
    
    if not args.output and not args.show:
        print("Error: Debes especificar --output o --show")
        return
    
    composer = MaskComposer(args.mask_path)
    
    try:
        composer.compose(
            target_width=args.width,
            output_path=args.output,
            show=args.show,
            add_legend=not args.no_legend
        )
    finally:
        composer.close()


if __name__ == "__main__":
    main()
