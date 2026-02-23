#!/usr/bin/env python3
"""
QuPath Mask Content Finder

Script que busca y muestra automáticamente regiones de la máscara
que contengan anotaciones (valores distintos de 0/background).

Uso:
    python view_mask_content.py /ruta/a/mascara.tif

Controles interactivos:
    - Flecha derecha / N: Siguiente región
    - Flecha izquierda / P: Región anterior
    - Q / Escape: Salir

Opciones:
    --size: Tamaño de la región a mostrar (default: 2048)
    --class-filter: Filtrar por clase específica
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import tifffile


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


class MaskContentFinder:
    """
    Encuentra y visualiza regiones de la máscara con contenido no-background.
    """
    
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
        print(f"Dimensiones: {self.width} x {self.height}")
        print(f"Tileado: {self.is_tiled}")
    
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
    
    def _load_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        width = min(width, self.width - x)
        height = min(height, self.height - y)
        
        if not self.is_tiled:
            full_data = self.page.asarray()
            return self._ensure_2d(full_data)[y:y+height, x:x+width]
        
        tile_x_start = x // self.tile_width
        tile_y_start = y // self.tile_height
        tile_x_end = (x + width - 1) // self.tile_width + 1
        tile_y_end = (y + height - 1) // self.tile_height + 1
        
        n_tiles_x = tile_x_end - tile_x_start
        n_tiles_y = tile_y_end - tile_y_start
        
        result_height = n_tiles_y * self.tile_height
        result_width = n_tiles_x * self.tile_width
        
        result = np.zeros((result_height, result_width), dtype=self.page.dtype)
        
        tiles_per_row = (self.width + self.tile_width - 1) // self.tile_width
        
        for iy, ty in enumerate(range(tile_y_start, tile_y_end)):
            for ix, tx in enumerate(range(tile_x_start, tile_x_end)):
                ry = iy * self.tile_height
                rx = ix * self.tile_width
                
                try:
                    tile_data = self._read_tile(tx, ty, tiles_per_row)
                    tile_2d = self._ensure_2d(tile_data)
                    th, tw = tile_2d.shape
                    
                    th = min(th, result_height - ry)
                    tw = min(tw, result_width - rx)
                    
                    result[ry:ry+th, rx:rx+tw] = tile_2d[:th, :tw]
                except Exception as e:
                    print(f"Error leyendo tile ({tx}, {ty}): {e}")
        
        offset_x = x - tile_x_start * self.tile_width
        offset_y = y - tile_y_start * self.tile_height
        
        return result[offset_y:offset_y+height, offset_x:offset_x+width]
    
    def find_all_content_regions(
        self, 
        region_size: int = 2048, 
        class_filter: int | None = None
    ) -> list[tuple[int, int, int, int, set[int]]]:
        """
        Busca todas las regiones que contengan contenido no-background.
        
        Args:
            region_size: Tamaño de la región a buscar
            class_filter: Si se especifica, busca solo esta clase
        
        Returns:
            Lista de tuplas (x, y, width, height, classes_found)
        """
        regions: list[tuple[int, int, int, int, set[int]]] = []
        seen_tiles: set[tuple[int, int]] = set()
        
        if not self.is_tiled:
            print("Cargando imagen completa para búsqueda...")
            full_data = self._ensure_2d(self.page.asarray())
            region = self._find_in_array(full_data, region_size, class_filter)
            if region:
                classes = set(np.unique(full_data)) - {0}
                regions.append((*region, classes))
            return regions
        
        tiles_per_row = (self.width + self.tile_width - 1) // self.tile_width
        tiles_per_col = (self.height + self.tile_height - 1) // self.tile_height
        total_tiles = tiles_per_row * tiles_per_col
        
        print(f"Buscando contenido en {total_tiles} tiles...")
        
        tiles_in_region = max(1, region_size // self.tile_width)
        
        for ty in range(tiles_per_col):
            for tx in range(tiles_per_row):
                if (tx, ty) in seen_tiles:
                    continue
                    
                try:
                    tile = self._read_tile(tx, ty, tiles_per_row)
                    tile_2d = self._ensure_2d(tile)
                    
                    if class_filter is not None:
                        has_content = np.any(tile_2d == class_filter)
                    else:
                        has_content = np.any(tile_2d != 0)
                    
                    if has_content:
                        if class_filter is not None:
                            mask = tile_2d == class_filter
                        else:
                            mask = tile_2d != 0
                        
                        positions = np.where(mask)
                        if len(positions[0]) > 0:
                            center_y_in_tile = int(np.mean(positions[0]))
                            center_x_in_tile = int(np.mean(positions[1]))
                            
                            global_x = tx * self.tile_width + center_x_in_tile
                            global_y = ty * self.tile_height + center_y_in_tile
                            
                            x = max(0, global_x - region_size // 2)
                            y = max(0, global_y - region_size // 2)
                            
                            if x + region_size > self.width:
                                x = self.width - region_size
                            if y + region_size > self.height:
                                y = self.height - region_size
                            
                            x = max(0, x)
                            y = max(0, y)
                            
                            w = min(region_size, self.width - x)
                            h = min(region_size, self.height - y)
                            
                            classes_in_tile = set(np.unique(tile_2d)) - {0}
                            regions.append((x, y, w, h, classes_in_tile))
                            
                            for dy in range(-tiles_in_region, tiles_in_region + 1):
                                for dx in range(-tiles_in_region, tiles_in_region + 1):
                                    seen_tiles.add((tx + dx, ty + dy))
                            
                except Exception:
                    continue
            
            progress = (ty + 1) / tiles_per_col * 100
            print(f"\rProgreso: {progress:.1f}% - Regiones encontradas: {len(regions)}", end="", flush=True)
        
        print()
        return regions
    
    def _find_in_array(
        self, 
        data: np.ndarray, 
        region_size: int, 
        class_filter: int | None
    ) -> tuple[int, int, int, int] | None:
        """Busca contenido en un array numpy."""
        if class_filter is not None:
            mask = data == class_filter
        else:
            mask = data != 0
        
        positions = np.where(mask)
        if len(positions[0]) == 0:
            return None
        
        center_y = int(np.mean(positions[0]))
        center_x = int(np.mean(positions[1]))
        
        x = max(0, center_x - region_size // 2)
        y = max(0, center_y - region_size // 2)
        
        h, w = data.shape
        if x + region_size > w:
            x = w - region_size
        if y + region_size > h:
            y = h - region_size
        
        x = max(0, x)
        y = max(0, y)
        
        final_w = min(region_size, w - x)
        final_h = min(region_size, h - y)
        
        return (x, y, final_w, final_h)
    
    def view_content(
        self, 
        region_size: int = 2048, 
        class_filter: int | None = None
    ) -> None:
        """
        Encuentra y visualiza todas las regiones con contenido.
        Permite navegar entre ellas con las teclas.
        
        Args:
            region_size: Tamaño de la región a mostrar
            class_filter: Filtrar por clase específica
        """
        filter_msg = f" (clase {class_filter}: {CLASS_NAMES.get(class_filter, '?')})" if class_filter else ""
        print(f"Buscando regiones con contenido{filter_msg}...")
        
        regions = self.find_all_content_regions(region_size, class_filter)
        
        if not regions:
            print("No se encontró contenido en la máscara.")
            if class_filter is not None:
                print(f"No hay anotaciones de clase {class_filter} ({CLASS_NAMES.get(class_filter, '?')}).")
            return
        
        print(f"\nEncontradas {len(regions)} regiones con contenido.")
        print("Controles: ← / P = Anterior | → / N = Siguiente | Q / Esc = Salir")
        
        viewer = RegionViewer(self, regions, class_filter)
        viewer.show()
    
    def list_classes(self) -> dict[int, int]:
        """
        Lista todas las clases presentes en la máscara con sus conteos.
        Escanea la imagen por tiles para no saturar memoria.
        
        Returns:
            Diccionario {class_id: pixel_count}
        """
        class_counts: dict[int, int] = {}
        
        if not self.is_tiled:
            print("Cargando imagen completa para análisis...")
            full_data = self._ensure_2d(self.page.asarray())
            unique, counts = np.unique(full_data, return_counts=True)
            return dict(zip(unique, counts))
        
        tiles_per_row = (self.width + self.tile_width - 1) // self.tile_width
        tiles_per_col = (self.height + self.tile_height - 1) // self.tile_height
        total_tiles = tiles_per_row * tiles_per_col
        
        print(f"Escaneando {total_tiles} tiles...")
        
        for i, ty in enumerate(range(tiles_per_col)):
            for tx in range(tiles_per_row):
                try:
                    tile = self._read_tile(tx, ty, tiles_per_row)
                    tile_2d = self._ensure_2d(tile)
                    unique, counts = np.unique(tile_2d, return_counts=True)
                    
                    for class_id, count in zip(unique, counts):
                        class_counts[class_id] = class_counts.get(class_id, 0) + int(count)
                except Exception:
                    continue
            
            progress = (i + 1) / tiles_per_col * 100
            print(f"\rProgreso: {progress:.1f}%", end="", flush=True)
        
        print()
        return class_counts
    
    def close(self):
        self.tif.close()


class RegionViewer:
    """Visor interactivo para navegar entre regiones."""
    
    def __init__(
        self, 
        finder: MaskContentFinder, 
        regions: list[tuple[int, int, int, int, set[int]]], 
        class_filter: int | None = None
    ):
        self.finder = finder
        self.regions = regions
        self.class_filter = class_filter
        self.current_idx = 0
        
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.im = None
        
        self.cmap = ListedColormap(CLASS_COLORS[:25])
        self.bounds = np.arange(-0.5, 25.5, 1)
        self.norm = BoundaryNorm(self.bounds, self.cmap.N)
    
    def _load_and_display(self) -> None:
        """Carga y muestra la región actual."""
        x, y, w, h, classes_hint = self.regions[self.current_idx]
        
        print(f"\nCargando región {self.current_idx + 1}/{len(self.regions)}: x={x}, y={y}")
        data = self.finder._load_region(x, y, w, h)
        
        unique_classes = np.unique(data)
        
        if self.im is None:
            self.im = self.ax.imshow(
                data,
                cmap=self.cmap,
                norm=self.norm,
                interpolation='nearest'
            )
        else:
            self.im.set_data(data)
        
        title = f"Región {self.current_idx + 1}/{len(self.regions)}: x={x}, y={y}, w={w}, h={h}"
        if self.class_filter is not None:
            title += f" (filtro: {CLASS_NAMES.get(self.class_filter, '?')})"
        
        self.ax.set_title(title)
        self.ax.set_xlabel(f"X (píxeles, offset={x})")
        self.ax.set_ylabel(f"Y (píxeles, offset={y})")
        
        legend_patches = [
            Patch(
                facecolor=CLASS_COLORS[class_id],
                edgecolor='black',
                label=f"{class_id}: {CLASS_NAMES[class_id]}"
            )
            for class_id in sorted(unique_classes) if class_id in CLASS_NAMES
        ]
        
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        
        self.ax.legend(
            handles=legend_patches,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            framealpha=0.9
        )
        
        self.fig.canvas.draw_idle()
        
        print(f"Clases en esta región: {[int(c) for c in sorted(unique_classes) if c != 0]}")
    
    def _on_key(self, event) -> None:
        """Maneja eventos de teclado."""
        if event.key in ['right', 'n', 'N']:
            if self.current_idx < len(self.regions) - 1:
                self.current_idx += 1
                self._load_and_display()
            else:
                print("Ya estás en la última región.")
        
        elif event.key in ['left', 'p', 'P']:
            if self.current_idx > 0:
                self.current_idx -= 1
                self._load_and_display()
            else:
                print("Ya estás en la primera región.")
        
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)
    
    def show(self) -> None:
        """Muestra el visor interactivo."""
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self._load_and_display()
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Encuentra y muestra regiones con contenido en máscaras de QuPath",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Mostrar primera región con contenido
  python view_mask_content.py mascara.tif
  
  # Mostrar región de 4096x4096
  python view_mask_content.py mascara.tif --size 4096
  
  # Buscar específicamente tumor (clase 24)
  python view_mask_content.py mascara.tif --class-filter 24
  
  # Listar todas las clases presentes
  python view_mask_content.py mascara.tif --list-classes
        """
    )
    
    parser.add_argument("mask_path", help="Ruta al archivo TIFF de la máscara")
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=2048,
        help="Tamaño de la región a mostrar (default: 2048)"
    )
    parser.add_argument(
        "--class-filter", "-c",
        type=int,
        default=None,
        help="Filtrar por clase específica (0-24)"
    )
    parser.add_argument(
        "--list-classes", "-l",
        action="store_true",
        help="Listar todas las clases presentes en la máscara"
    )
    
    args = parser.parse_args()
    
    finder = MaskContentFinder(args.mask_path)
    
    try:
        if args.list_classes:
            print("\nAnalizando clases en la máscara...")
            counts = finder.list_classes()
            
            print("\nClases encontradas:")
            print("-" * 60)
            total_pixels = sum(counts.values())
            
            for class_id in sorted(counts.keys()):
                count = counts[class_id]
                percentage = 100.0 * count / total_pixels
                class_name = CLASS_NAMES.get(class_id, "Desconocida")
                print(f"  {class_id:2d}: {class_name:30s} - {count:12d} px ({percentage:6.3f}%)")
            
            print("-" * 60)
            print(f"Total: {total_pixels:d} píxeles")
        else:
            finder.view_content(
                region_size=args.size,
                class_filter=args.class_filter
            )
    finally:
        finder.close()


if __name__ == "__main__":
    main()
