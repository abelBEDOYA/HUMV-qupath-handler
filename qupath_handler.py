#!/usr/bin/env python3
"""
QuPath Export Handler

Clase para manejar eficientemente las imágenes y máscaras OME-TIFF piramidales
exportadas desde QuPath. Optimizada para no saturar RAM.

Características:
- Lectura eficiente usando niveles de pirámide
- Visualización interactiva con zoom/pan sincronizado
- Carga lazy: solo carga lo que se muestra
- Soporte para lectura por regiones

Uso:
    from qupath_handler import QuPathHandler
    
    handler = QuPathHandler("/ruta/a/datos")
    handler.load_pair("nombre_imagen")
    handler.visualize_interactive()
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
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

CLASS_COLORS_HEX = [
    "#000000",  # 0: Background
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


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convierte color hex a RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


CLASS_COLORS_RGB = [hex_to_rgb(c) for c in CLASS_COLORS_HEX]


class PyramidTiff:
    """
    Wrapper eficiente para TIFF piramidal (OME-TIFF).
    
    Detecta automáticamente la estructura de la pirámide:
    - series[0].levels: Niveles dentro de una serie (OME-TIFF estándar)
    - Múltiples series: Cada serie es un nivel
    - Páginas: Cada página es un nivel
    """
    
    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.tif = tifffile.TiffFile(path)
        
        # Detectar estructura de la pirámide
        self._detect_pyramid_structure()
        
        # Cachear info de niveles
        self._cache_level_info()
        
        if self.verbose:
            self._print_info()
    
    def _detect_pyramid_structure(self) -> None:
        """Detecta cómo están organizados los niveles de la pirámide."""
        self._pyramid_type = None
        self._levels_source = []
        
        # Opción 1: series[0].levels (OME-TIFF con subresoluciones)
        if len(self.tif.series) > 0:
            first_series = self.tif.series[0]
            if hasattr(first_series, 'levels') and len(first_series.levels) > 1:
                self._pyramid_type = 'series_levels'
                self._levels_source = first_series.levels
                self.n_levels = len(self._levels_source)
                return
        
        # Opción 2: Múltiples series (cada una es un nivel)
        if len(self.tif.series) > 1:
            # Verificar que las series son de tamaño decreciente
            shapes = [s.shape for s in self.tif.series]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'multiple_series'
                self._levels_source = self.tif.series
                self.n_levels = len(self._levels_source)
                return
        
        # Opción 3: Múltiples páginas
        if len(self.tif.pages) > 1:
            shapes = [p.shape for p in self.tif.pages]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'pages'
                self._levels_source = list(self.tif.pages)
                self.n_levels = len(self._levels_source)
                return
        
        # Fallback: Solo un nivel
        self._pyramid_type = 'single'
        if len(self.tif.series) > 0:
            self._levels_source = [self.tif.series[0]]
        else:
            self._levels_source = [self.tif.pages[0]]
        self.n_levels = 1
    
    def _shapes_are_pyramid(self, shapes: List[Tuple]) -> bool:
        """Verifica si las shapes corresponden a una pirámide (tamaños decrecientes)."""
        if len(shapes) < 2:
            return False
        
        # Extraer el tamaño principal de cada shape
        sizes = []
        for shape in shapes:
            # Tomar las dos dimensiones más grandes
            dims = sorted(shape, reverse=True)[:2]
            sizes.append(max(dims))
        
        # Verificar que son decrecientes
        for i in range(1, len(sizes)):
            if sizes[i] >= sizes[i-1]:
                return False
        return True
    
    def _cache_level_info(self) -> None:
        """Cachea información de cada nivel SIN cargar datos."""
        self.level_info: List[Dict[str, Any]] = []
        base_w, base_h = 0, 0
        
        for i, level_src in enumerate(self._levels_source):
            shape = level_src.shape
            h, w, c = self._parse_shape(shape)
            
            if i == 0:
                ds = 1.0
                base_w, base_h = w, h
            else:
                ds = base_w / w if w > 0 else 1.0
            
            self.level_info.append({
                'index': i,
                'shape': shape,
                'width': w,
                'height': h,
                'channels': c,
                'downsample': ds
            })
    
    def _parse_shape(self, shape: Tuple) -> Tuple[int, int, int]:
        """Interpreta shape para obtener H, W, C."""
        if len(shape) == 2:
            return shape[0], shape[1], 1
        elif len(shape) == 3:
            if shape[0] <= 4:  # (C, H, W)
                return shape[1], shape[2], shape[0]
            else:  # (H, W, C)
                return shape[0], shape[1], shape[2]
        elif len(shape) >= 4:
            # OME: (T, C, Z, Y, X) o similar - Y, X son las últimas
            return shape[-2], shape[-1], shape[1] if shape[1] <= 4 else 1
        return shape[0], 1, 1
    
    def _print_info(self) -> None:
        """Imprime información de debug."""
        print(f"  [PyramidTiff] Tipo: {self._pyramid_type}")
        print(f"  [PyramidTiff] Niveles: {self.n_levels}")
        for info in self.level_info:
            print(f"    Nivel {info['index']}: {info['width']}x{info['height']} "
                  f"(ds: {info['downsample']:.0f}x, shape: {info['shape']})")
    
    def get_level_for_display(self, max_pixels: int = 4_000_000) -> int:
        """Devuelve el nivel más adecuado dado un límite de píxeles."""
        for info in self.level_info:
            pixels = info['width'] * info['height']
            if pixels <= max_pixels:
                return info['index']
        return self.n_levels - 1
    
    def read_level(self, level: int = 0) -> np.ndarray:
        """Lee un nivel completo de la pirámide."""
        level = min(level, self.n_levels - 1)
        
        if self.verbose:
            info = self.level_info[level]
            print(f"  [PyramidTiff] Leyendo nivel {level}: {info['width']}x{info['height']}")
        
        level_src = self._levels_source[level]
        data = level_src.asarray()
        
        if self.verbose:
            print(f"  [PyramidTiff] Cargado: shape={data.shape}, RAM={data.nbytes/1024/1024:.1f}MB")
        
        return self._normalize_shape(data)
    
    def _normalize_shape(self, data: np.ndarray) -> np.ndarray:
        """Normaliza a (H, W) o (H, W, C)."""
        data = np.squeeze(data)
        
        if data.ndim == 2:
            return data
        elif data.ndim == 3:
            if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, -1)
            return data
        
        # Más de 3 dimensiones: reducir
        while data.ndim > 3:
            data = data[0]
        return self._normalize_shape(data)
    
    @property
    def base_shape(self) -> Tuple[int, int]:
        """Devuelve (width, height) del nivel base."""
        return (self.level_info[0]['width'], self.level_info[0]['height'])
    
    def close(self) -> None:
        self.tif.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class QuPathHandler:
    """
    Manejador para pares imagen/máscara exportados desde QuPath.
    Optimizado para visualización eficiente sin saturar RAM.
    """
    
    def __init__(
        self, 
        data_dir: str,
        images_subdir: str = "images1",
        masks_subdir: str = "masks1"
    ):
        """
        Args:
            data_dir: Directorio base con los datos
            images_subdir: Subdirectorio de imágenes (o mismo dir si es None)
            masks_subdir: Subdirectorio de máscaras (o mismo dir si es None)
        """
        self.data_dir = Path(data_dir)
        
        if images_subdir:
            self.images_dir = self.data_dir / images_subdir
        else:
            self.images_dir = self.data_dir
            
        if masks_subdir:
            self.masks_dir = self.data_dir / masks_subdir
        else:
            self.masks_dir = self.data_dir
        
        # Estado actual
        self.current_name: str | None = None
        self.image_tiff: PyramidTiff | None = None
        self.mask_tiff: PyramidTiff | None = None
        self.current_level: int = 0
        
        # Datos cargados del nivel actual
        self.image_data: np.ndarray | None = None
        self.mask_data: np.ndarray | None = None
        
        # Colormap para máscaras
        self.mask_cmap = ListedColormap(CLASS_COLORS_HEX[:25])
        self.mask_norm = BoundaryNorm(np.arange(-0.5, 25.5, 1), self.mask_cmap.N)
    
    def list_images(self) -> List[str]:
        """Lista todas las imágenes disponibles."""
        patterns = ["*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff", "*__mask_multiclass.ome.tif"]
        images = []
        for pattern in patterns:
            images.extend(self.images_dir.glob(pattern))
        
        # Extraer nombres base (sin extensión ni sufijos)
        names = set()
        for img in images:
            name = img.stem
            # Quitar sufijos comunes
            for suffix in [".ome", "__mask_multiclass"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            names.add(name)
        
        return sorted(names)
    
    def _find_file(self, directory: Path, base_name: str, suffixes: List[str]) -> Path | None:
        """Busca un archivo con diferentes posibles nombres."""
        extensions = [".ome.tif", ".ome.tiff", ".tif", ".tiff"]
        
        for suffix in suffixes:
            for ext in extensions:
                path = directory / f"{base_name}{suffix}{ext}"
                if path.exists():
                    return path
                print(path)
        return None
    
    def load_pair(self, name: str, level: int | None = None) -> None:
        """
        Carga un par imagen/máscara.
        
        Args:
            name: Nombre base de la imagen (sin extensión)
            level: Nivel de pirámide a cargar. None = auto (nivel que quepa en ~4MP)
        """
        # Cerrar archivos anteriores
        self.close()
        
        self.current_name = name
        
        # Buscar imagen
        image_path = self._find_file(
            self.images_dir, 
            name, 
            ["", ".ome"]
        )
        
        # Buscar máscara
        mask_path = self._find_file(
            self.masks_dir, 
            name, 
            ["__mask_multiclass", "_mask", "__mask"]
        )
        
        if image_path is None:
            print(f"No se encontró imagen para: {name}")
            print(f"Buscado en: {self.images_dir}")
            return
        
        print(f"Cargando imagen: {image_path.name}")
        self.image_tiff = PyramidTiff(str(image_path))
        
        if mask_path is not None:
            print(f"Cargando máscara: {mask_path.name}")
            self.mask_tiff = PyramidTiff(str(mask_path))
        else:
            print(f"No se encontró máscara para: {name}")
            self.mask_tiff = None
        
        # Determinar nivel a cargar
        if level is None:
            level = self.image_tiff.get_level_for_display(max_pixels=4_000_000)
        
        self._load_level(level)
    
    def _load_level(self, level: int) -> None:
        """Carga un nivel específico en memoria."""
        import gc
        
        if self.image_tiff is None:
            return
        
        level = min(level, self.image_tiff.n_levels - 1)
        self.current_level = level
        
        print(f"\nCargando nivel {level}...")
        
        # Liberar memoria del nivel anterior ANTES de cargar el nuevo
        self.image_data = None
        self.mask_data = None
        gc.collect()
        
        # Cargar imagen
        self.image_data = self.image_tiff.read_level(level)
        info = self.image_tiff.level_info[level]
        print(f"  Imagen: {info['width']} x {info['height']} (downsample: {info['downsample']:.1f}x)")
        print(f"  RAM imagen: {self.image_data.nbytes / 1024 / 1024:.1f} MB")
        
        # Cargar máscara al mismo nivel si existe
        if self.mask_tiff is not None:
            mask_level = min(level, self.mask_tiff.n_levels - 1)
            self.mask_data = self.mask_tiff.read_level(mask_level)
            
            # Asegurar 2D para máscara
            if self.mask_data.ndim == 3:
                self.mask_data = self.mask_data[:, :, 0]
            
            print(f"  Máscara: {self.mask_data.shape}")
            print(f"  RAM máscara: {self.mask_data.nbytes / 1024 / 1024:.1f} MB")
        
        gc.collect()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve metadatos del par actual."""
        if self.image_tiff is None:
            return {}
        
        return {
            'name': self.current_name,
            'current_level': self.current_level,
            'image_levels': [
                {
                    'level': i,
                    'size': f"{info['width']} x {info['height']}",
                    'downsample': info['downsample']
                }
                for i, info in enumerate(self.image_tiff.level_info)
            ],
            'mask_levels': [
                {
                    'level': i,
                    'size': f"{info['width']} x {info['height']}",
                    'downsample': info['downsample']
                }
                for i, info in enumerate(self.mask_tiff.level_info)
            ] if self.mask_tiff else [],
            'base_size': self.image_tiff.base_shape,
        }
    
    def get_data(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Devuelve (imagen, máscara) del nivel actual."""
        return self.image_data, self.mask_data
    
    def visualize(self, show_legend: bool = True) -> None:
        """
        Visualización con zoom/pan sincronizado entre imagen y máscara.
        Usa la herramienta de zoom de matplotlib para explorar.
        """
        if self.image_data is None:
            print("No hay datos cargados. Usa load_pair() primero.")
            return
        
        has_mask = self.mask_data is not None
        n_cols = 2 if has_mask else 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
        if n_cols == 1:
            axes = [axes]
        
        ax_img = axes[0]
        ax_mask = axes[1] if has_mask else None
        
        # Título
        info = self.image_tiff.level_info[self.current_level]
        title = f"{self.current_name} | Level {self.current_level} | {info['width']}x{info['height']} (ds: {info['downsample']:.0f}x)"
        fig.suptitle(title, fontsize=12)
        
        # Imagen
        ax_img.imshow(self.image_data)
        ax_img.set_title("Imagen")
        ax_img.axis('off')
        
        # Máscara
        if has_mask and ax_mask is not None:
            ax_mask.imshow(
                self.mask_data,
                cmap=self.mask_cmap,
                norm=self.mask_norm,
                interpolation='nearest'
            )
            ax_mask.set_title("Máscara")
            ax_mask.axis('off')
            
            if show_legend:
                unique_classes = np.unique(self.mask_data)
                legend_patches = [
                    Patch(
                        facecolor=CLASS_COLORS_HEX[c],
                        edgecolor='black',
                        label=f"{c}: {CLASS_NAMES.get(c, '?')}"
                    )
                    for c in sorted(unique_classes) if c in CLASS_NAMES
                ]
                
                if len(legend_patches) <= 10:
                    ax_mask.legend(
                        handles=legend_patches,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=8
                    )
            
            # Sincronizar zoom/pan entre los dos axes
            self._syncing = False
            
            def sync_from_img(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_mask.set_xlim(ax_img.get_xlim())
                    ax_mask.set_ylim(ax_img.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            def sync_from_mask(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_img.set_xlim(ax_mask.get_xlim())
                    ax_img.set_ylim(ax_mask.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            ax_img.callbacks.connect('xlim_changed', sync_from_img)
            ax_img.callbacks.connect('ylim_changed', sync_from_img)
            ax_mask.callbacks.connect('xlim_changed', sync_from_mask)
            ax_mask.callbacks.connect('ylim_changed', sync_from_mask)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_interactive(self) -> None:
        """
        Visualización interactiva con slider de nivel.
        Permite cambiar el nivel de resolución en tiempo real.
        """
        if self.image_tiff is None:
            print("No hay datos cargados. Usa load_pair() primero.")
            return
        
        has_mask = self.mask_data is not None
        n_cols = 2 if has_mask else 1
        
        # Crear figura con espacio para slider
        fig = plt.figure(figsize=(7 * n_cols, 8))
        
        # Axes para las imágenes
        ax_img = fig.add_axes([0.05, 0.15, 0.4 if has_mask else 0.9, 0.75])
        ax_mask = fig.add_axes([0.55, 0.15, 0.4, 0.75]) if has_mask else None
        
        # Axis para slider
        ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
        
        # Slider de nivel
        n_levels = self.image_tiff.n_levels
        slider = Slider(
            ax_slider,
            'Nivel',
            0,
            n_levels - 1,
            valinit=self.current_level,
            valstep=1
        )
        
        # Mostrar imágenes iniciales
        img_display = ax_img.imshow(self.image_data)
        ax_img.set_title("Imagen")
        ax_img.axis('off')
        
        mask_display = None
        if has_mask and ax_mask is not None:
            mask_display = ax_mask.imshow(
                self.mask_data,
                cmap=self.mask_cmap,
                norm=self.mask_norm,
                interpolation='nearest'
            )
            ax_mask.set_title("Máscara")
            ax_mask.axis('off')
        
        def update_title():
            info = self.image_tiff.level_info[self.current_level]
            fig.suptitle(
                f"{self.current_name} | Level {self.current_level} | "
                f"{info['width']}x{info['height']} (ds: {info['downsample']:.0f}x)",
                fontsize=11
            )
        
        update_title()
        
        def on_slider_change(val):
            level = int(val)
            if level != self.current_level:
                self._load_level(level)
                img_display.set_data(self.image_data)
                img_display.set_extent([0, self.image_data.shape[1], self.image_data.shape[0], 0])
                ax_img.set_xlim(0, self.image_data.shape[1])
                ax_img.set_ylim(self.image_data.shape[0], 0)
                
                if mask_display is not None and self.mask_data is not None:
                    mask_display.set_data(self.mask_data)
                    mask_display.set_extent([0, self.mask_data.shape[1], self.mask_data.shape[0], 0])
                    ax_mask.set_xlim(0, self.mask_data.shape[1])
                    ax_mask.set_ylim(self.mask_data.shape[0], 0)
                
                update_title()
                fig.canvas.draw_idle()
        
        slider.on_changed(on_slider_change)
        
        # Sincronizar zoom/pan entre los dos axes
        if ax_mask is not None:
            # Flag para evitar recursión infinita
            self._syncing = False
            
            def sync_from_img(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_mask.set_xlim(ax_img.get_xlim())
                    ax_mask.set_ylim(ax_img.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            def sync_from_mask(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_img.set_xlim(ax_mask.get_xlim())
                    ax_img.set_ylim(ax_mask.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            ax_img.callbacks.connect('xlim_changed', sync_from_img)
            ax_img.callbacks.connect('ylim_changed', sync_from_img)
            ax_mask.callbacks.connect('xlim_changed', sync_from_mask)
            ax_mask.callbacks.connect('ylim_changed', sync_from_mask)
        
        plt.show()
    
    def change_level(self, level: int) -> None:
        """Cambia al nivel especificado."""
        if self.image_tiff is None:
            print("No hay datos cargados.")
            return
        self._load_level(level)
    
    def close(self) -> None:
        """Cierra los archivos abiertos y libera memoria."""
        if self.image_tiff is not None:
            self.image_tiff.close()
            self.image_tiff = None
        
        if self.mask_tiff is not None:
            self.mask_tiff.close()
            self.mask_tiff = None
        
        self.image_data = None
        self.mask_data = None
        self.current_name = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def main():
    """Ejemplo de uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualizador de imágenes y máscaras QuPath",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Visualizar un par imagen/máscara
  python qupath_handler.py /ruta/datos --name imagen1
  
  # Listar imágenes disponibles
  python qupath_handler.py /ruta/datos --list
  
  # Especificar nivel de resolución
  python qupath_handler.py /ruta/datos --name imagen1 --level 2

Estructura de directorios esperada:
  data_dir/
    images/
      imagen1.ome.tif
      imagen2.ome.tif
    masks/
      imagen1__mask_multiclass.ome.tif
      imagen2__mask_multiclass.ome.tif
        """
    )
    
    parser.add_argument("data_dir", help="Directorio con los datos")
    parser.add_argument("--name", "-n", help="Nombre de la imagen a visualizar")
    parser.add_argument("--level", "-l", type=int, default=None, help="Nivel de pirámide")
    parser.add_argument("--list", action="store_true", help="Listar imágenes disponibles")
    parser.add_argument("--images-dir", default="images1", help="Subdirectorio de imágenes")
    parser.add_argument("--masks-dir", default="masks1", help="Subdirectorio de máscaras")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo con slider")
    
    args = parser.parse_args()
    
    handler = QuPathHandler(
        args.data_dir,
        images_subdir=args.images_dir,
        masks_subdir=args.masks_dir
    )
    
    if args.list:
        images = handler.list_images()
        print(f"\nImágenes encontradas ({len(images)}):")
        for img in images:
            print(f"  - {img}")
        return
    
    if args.name is None:
        images = handler.list_images()
        if images:
            args.name = images[0]
            print(f"Usando primera imagen: {args.name}")
        else:
            print("No se encontraron imágenes.")
            return
    
    try:
        handler.load_pair(args.name, level=args.level)
        
        print("\nMetadatos:")
        meta = handler.get_metadata()
        print(f"  Nombre: {meta.get('name')}")
        print(f"  Tamaño base: {meta.get('base_size')}")
        print(f"  Niveles imagen: {len(meta.get('image_levels', []))}")
        for lvl in meta.get('image_levels', []):
            print(f"    {lvl['level']}: {lvl['size']} (ds: {lvl['downsample']:.0f}x)")
        
        if args.interactive:
            handler.visualize_interactive()
        else:
            handler.visualize()
    finally:
        handler.close()


if __name__ == "__main__":
    main()
