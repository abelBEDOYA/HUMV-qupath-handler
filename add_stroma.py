#!/usr/bin/env python3
"""
Add Stroma to Masks

Script para detectar tejido en imágenes histológicas mediante thresholding flexible
y marcar como Stroma (clase 23) las zonas de tejido sin anotación previa.

Genera TIFFs piramidales coherentes con la estructura original.

Uso:
    python add_stroma.py /path/to/dataset
    python add_stroma.py /path/to/dataset --blur 7 --threshold 235
    python add_stroma.py /path/to/dataset --preview
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Generator

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing


STROMA_CLASS_ID = 23
BACKGROUND_CLASS_ID = 0

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


class TissueDetector:
    """
    Detecta zonas con tejido en tiles de imágenes histológicas.
    
    Usa thresholding flexible con desenfoque y operaciones morfológicas
    para obtener máscaras de tejido suaves y conectadas.
    """
    
    def __init__(
        self,
        blur_radius: float = 5.0,
        threshold: int = 240,
        min_area: int = 1000,
        dilate_radius: int = 10
    ):
        """
        Args:
            blur_radius: Radio del desenfoque gaussiano (mayor = más suave)
            threshold: Umbral de blancura (0-255, menor = más sensible)
            min_area: Área mínima de región para considerar (píxeles)
            dilate_radius: Radio de dilatación para unir zonas cercanas
        """
        self.blur_radius = blur_radius
        self.threshold = threshold
        self.min_area = min_area
        self.dilate_radius = dilate_radius
    
    def detect(self, rgb_tile: np.ndarray) -> np.ndarray:
        """
        Detecta tejido en un tile RGB.
        
        Args:
            rgb_tile: Array (H, W, 3) con valores 0-255
            
        Returns:
            Máscara booleana (H, W) donde True = tejido detectado
        """
        if rgb_tile.ndim == 2:
            gray = rgb_tile.astype(np.float32)
        else:
            gray = np.mean(rgb_tile.astype(np.float32), axis=2)
        
        if self.blur_radius > 0:
            blurred = gaussian_filter(gray, sigma=self.blur_radius)
        else:
            blurred = gray
        
        tissue_mask = blurred < self.threshold
        
        if self.dilate_radius > 0:
            struct = np.ones((self.dilate_radius * 2 + 1, self.dilate_radius * 2 + 1), dtype=bool)
            tissue_mask = binary_closing(tissue_mask, structure=struct)
            tissue_mask = binary_dilation(tissue_mask, structure=struct, iterations=1)
            tissue_mask = binary_erosion(tissue_mask, structure=struct, iterations=1)
        
        if self.min_area > 0:
            tissue_mask = self._remove_small_regions(tissue_mask, self.min_area)
        
        return tissue_mask
    
    def _remove_small_regions(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Elimina regiones pequeñas de la máscara."""
        from scipy.ndimage import label
        
        labeled, num_features = label(mask)
        if num_features == 0:
            return mask
        
        component_sizes = np.bincount(labeled.ravel())
        too_small = component_sizes < min_size
        too_small[0] = False
        
        mask_cleaned = mask.copy()
        mask_cleaned[too_small[labeled]] = False
        
        return mask_cleaned


class PyramidReader:
    """Lee TIFFs piramidales por tiles."""
    
    def __init__(self, path: str):
        self.path = path
        self.tif = tifffile.TiffFile(path)
        self._detect_structure()
    
    def _detect_structure(self) -> None:
        """Detecta la estructura de la pirámide."""
        if len(self.tif.series) > 0:
            first_series = self.tif.series[0]
            if hasattr(first_series, 'levels') and len(first_series.levels) > 1:
                self._levels = first_series.levels
                self.n_levels = len(self._levels)
                return
        
        if len(self.tif.series) > 1:
            self._levels = self.tif.series
            self.n_levels = len(self._levels)
            return
        
        self._levels = [self.tif.series[0]] if self.tif.series else [self.tif.pages[0]]
        self.n_levels = 1
    
    def get_level_shape(self, level: int = 0) -> Tuple[int, int]:
        """Devuelve (height, width) del nivel especificado."""
        shape = self._levels[level].shape
        if len(shape) == 2:
            return shape[0], shape[1]
        elif len(shape) == 3:
            if shape[0] <= 4:
                return shape[1], shape[2]
            return shape[0], shape[1]
        return shape[-2], shape[-1]
    
    def get_downsamples(self) -> List[float]:
        """Calcula los factores de downsample de cada nivel."""
        base_h, base_w = self.get_level_shape(0)
        downsamples = []
        for i in range(self.n_levels):
            h, w = self.get_level_shape(i)
            ds = base_w / w if w > 0 else 1.0
            downsamples.append(ds)
        return downsamples
    
    def read_level(self, level: int = 0) -> np.ndarray:
        """Lee un nivel completo."""
        data = self._levels[level].asarray()
        return self._normalize(data)
    
    def read_region(self, x: int, y: int, width: int, height: int, level: int = 0) -> np.ndarray:
        """Lee una región de un nivel."""
        level_h, level_w = self.get_level_shape(level)
        
        x = max(0, min(x, level_w))
        y = max(0, min(y, level_h))
        width = min(width, level_w - x)
        height = min(height, level_h - y)
        
        if width <= 0 or height <= 0:
            return np.zeros((height, width), dtype=np.uint8)
        
        store = self._levels[level].aszarr()
        
        try:
            import zarr
            z = zarr.open(store, mode='r')
            
            if z.ndim == 2:
                data = z[y:y+height, x:x+width]
            elif z.ndim == 3:
                if z.shape[0] <= 4:
                    data = z[:, y:y+height, x:x+width]
                    data = np.moveaxis(data, 0, -1)
                else:
                    data = z[y:y+height, x:x+width, :]
            else:
                data = z[..., y:y+height, x:x+width]
                data = np.squeeze(data)
            
            return np.asarray(data)
        except Exception:
            full_data = self.read_level(level)
            return full_data[y:y+height, x:x+width]
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normaliza datos a (H, W) o (H, W, C)."""
        data = np.squeeze(data)
        if data.ndim == 3 and data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
            data = np.moveaxis(data, 0, -1)
        return data
    
    def close(self) -> None:
        self.tif.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class StromaAdder:
    """
    Añade clase Stroma a máscaras en zonas de tejido sin anotación.
    
    Procesa imágenes grandes por tiles para no saturar RAM y genera
    TIFFs piramidales coherentes.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str | None = None,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        tile_size: int = 2048,
        detector_params: Dict[str, Any] | None = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / images_subdir
        self.masks_dir = self.dataset_dir / masks_subdir
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.dataset_dir / "masks_with_stroma"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tile_size = tile_size
        
        params = detector_params or {}
        self.detector = TissueDetector(**params)
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Encuentra pares imagen-máscara."""
        pairs = []
        
        for img_path in sorted(self.images_dir.glob("*.ome.tif")):
            base_name = img_path.stem
            if base_name.endswith(".ome"):
                base_name = base_name[:-4]
            
            mask_patterns = [
                f"{base_name}__mask_multiclass.ome.tif",
                f"{base_name}_mask.ome.tif",
                f"{base_name}__mask.ome.tif",
            ]
            
            mask_path = None
            for pattern in mask_patterns:
                candidate = self.masks_dir / pattern
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                pairs.append((img_path, mask_path))
            else:
                print(f"  Advertencia: No se encontró máscara para {img_path.name}")
        
        return pairs
    
    def process_all(self, specific_name: str | None = None) -> None:
        """Procesa todos los pares imagen-máscara."""
        pairs = self.find_pairs()
        
        if specific_name:
            pairs = [(i, m) for i, m in pairs if specific_name in i.stem]
        
        print(f"Encontrados {len(pairs)} pares imagen-máscara")
        print(f"Salida: {self.output_dir}")
        print(f"Parámetros de detección:")
        print(f"  - blur_radius: {self.detector.blur_radius}")
        print(f"  - threshold: {self.detector.threshold}")
        print(f"  - min_area: {self.detector.min_area}")
        print(f"  - dilate_radius: {self.detector.dilate_radius}")
        print()
        
        for i, (img_path, mask_path) in enumerate(pairs):
            print(f"[{i+1}/{len(pairs)}] Procesando: {img_path.name}")
            try:
                self.process_pair(img_path, mask_path)
                gc.collect()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    def process_pair(self, image_path: Path, mask_path: Path) -> None:
        """Procesa un par imagen-máscara."""
        with PyramidReader(str(image_path)) as img_reader, \
             PyramidReader(str(mask_path)) as mask_reader:
            
            img_h, img_w = img_reader.get_level_shape(0)
            mask_h, mask_w = mask_reader.get_level_shape(0)
            
            print(f"  Imagen: {img_w} x {img_h}")
            print(f"  Máscara: {mask_w} x {mask_h}")
            
            downsamples = mask_reader.get_downsamples()
            print(f"  Niveles: {len(downsamples)} ({', '.join(f'{d:.0f}x' for d in downsamples)})")
            
            print(f"  Procesando nivel 0 por tiles ({self.tile_size}x{self.tile_size})...")
            level0_mask = self._process_level0(img_reader, mask_reader)
            
            print(f"  Generando pirámide...")
            levels = self._generate_pyramid(level0_mask, downsamples)
            
            output_name = mask_path.name
            output_path = self.output_dir / output_name
            
            print(f"  Escribiendo {output_path.name}...")
            self._write_pyramid_tiff(output_path, levels)
            
            del level0_mask
            del levels
            gc.collect()
            
            file_size = output_path.stat().st_size
            print(f"  Completado: {file_size / 1024 / 1024:.1f} MB")
    
    def _process_level0(
        self,
        img_reader: PyramidReader,
        mask_reader: PyramidReader
    ) -> np.ndarray:
        """Procesa el nivel 0 por tiles y devuelve la máscara modificada."""
        mask_h, mask_w = mask_reader.get_level_shape(0)
        img_h, img_w = img_reader.get_level_shape(0)
        
        scale_x = img_w / mask_w
        scale_y = img_h / mask_h
        
        output_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        
        stroma_pixels_added = 0
        total_tiles = ((mask_h + self.tile_size - 1) // self.tile_size) * \
                      ((mask_w + self.tile_size - 1) // self.tile_size)
        tile_count = 0
        
        for y in range(0, mask_h, self.tile_size):
            for x in range(0, mask_w, self.tile_size):
                tile_count += 1
                
                tile_h = min(self.tile_size, mask_h - y)
                tile_w = min(self.tile_size, mask_w - x)
                
                mask_tile = mask_reader.read_region(x, y, tile_w, tile_h, level=0)
                if mask_tile.ndim == 3:
                    mask_tile = mask_tile[:, :, 0]
                
                img_x = int(x * scale_x)
                img_y = int(y * scale_y)
                img_tile_w = int(tile_w * scale_x)
                img_tile_h = int(tile_h * scale_y)
                
                img_tile = img_reader.read_region(img_x, img_y, img_tile_w, img_tile_h, level=0)
                
                if img_tile.shape[:2] != mask_tile.shape[:2]:
                    from scipy.ndimage import zoom
                    if img_tile.ndim == 3:
                        factors = (tile_h / img_tile.shape[0], tile_w / img_tile.shape[1], 1)
                    else:
                        factors = (tile_h / img_tile.shape[0], tile_w / img_tile.shape[1])
                    img_tile = zoom(img_tile, factors, order=1)
                
                tissue_mask = self.detector.detect(img_tile)
                
                stroma_candidates = tissue_mask & (mask_tile == BACKGROUND_CLASS_ID)
                
                mask_tile[stroma_candidates] = STROMA_CLASS_ID
                stroma_pixels_added += np.sum(stroma_candidates)
                
                output_mask[y:y+tile_h, x:x+tile_w] = mask_tile[:tile_h, :tile_w]
                
                if tile_count % 10 == 0 or tile_count == total_tiles:
                    print(f"    Tiles: {tile_count}/{total_tiles} "
                          f"({100*tile_count/total_tiles:.0f}%)", end='\r')
        
        print(f"    Tiles: {total_tiles}/{total_tiles} (100%) - "
              f"Píxeles stroma añadidos: {stroma_pixels_added:,}")
        
        return output_mask
    
    def _generate_pyramid(
        self,
        level0: np.ndarray,
        downsamples: List[float]
    ) -> List[np.ndarray]:
        """Genera niveles de pirámide desde el nivel 0."""
        from scipy.ndimage import zoom
        
        levels = [level0]
        h, w = level0.shape
        
        for ds in downsamples[1:]:
            new_h = int(h / ds)
            new_w = int(w / ds)
            
            factor = 1.0 / ds
            level_n = zoom(level0, factor, order=0, mode='nearest')
            
            if level_n.shape != (new_h, new_w):
                level_n = level_n[:new_h, :new_w]
            
            levels.append(level_n.astype(np.uint8))
            print(f"    Nivel ds={ds:.0f}x: {new_w} x {new_h}")
        
        return levels
    
    def _write_pyramid_tiff(
        self,
        output_path: Path,
        levels: List[np.ndarray],
        tile_size: int = 512
    ) -> None:
        """Escribe los niveles como OME-TIFF piramidal."""
        with tifffile.TiffWriter(str(output_path), ome=True, bigtiff=True) as tif:
            options = {
                'tile': (tile_size, tile_size),
                'compression': 'lzw',
                'photometric': 'minisblack',
            }
            
            tif.write(
                levels[0],
                subifds=len(levels) - 1,
                **options
            )
            
            for level in levels[1:]:
                tif.write(
                    level,
                    subfiletype=1,
                    **options
                )


def preview_thresholding(
    image_path: str,
    mask_path: str,
    detector: TissueDetector,
    level: int = 2
) -> None:
    """Muestra un preview del thresholding para ajustar parámetros."""
    import matplotlib.pyplot as plt
    
    print(f"Cargando preview (nivel {level})...")
    
    with PyramidReader(image_path) as img_reader, \
         PyramidReader(mask_path) as mask_reader:
        
        preview_level = min(level, img_reader.n_levels - 1, mask_reader.n_levels - 1)
        
        img_data = img_reader.read_level(preview_level)
        mask_data = mask_reader.read_level(preview_level)
        
        if mask_data.ndim == 3:
            mask_data = mask_data[:, :, 0]
        
        print(f"  Imagen preview: {img_data.shape}")
        print(f"  Máscara preview: {mask_data.shape}")
        
        tissue_mask = detector.detect(img_data)
        
        stroma_candidates = tissue_mask & (mask_data == BACKGROUND_CLASS_ID)
        
        preview_mask = mask_data.copy()
        preview_mask[stroma_candidates] = STROMA_CLASS_ID
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        axes[0, 0].imshow(img_data)
        axes[0, 0].set_title("Imagen original")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(tissue_mask, cmap='gray')
        axes[0, 1].set_title(f"Tejido detectado\n(blur={detector.blur_radius}, "
                            f"thresh={detector.threshold}, dilate={detector.dilate_radius})")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(mask_data, cmap='tab20', vmin=0, vmax=24)
        axes[1, 0].set_title("Máscara original")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(preview_mask, cmap='tab20', vmin=0, vmax=24)
        axes[1, 1].set_title(f"Máscara con stroma añadido\n"
                            f"(píxeles nuevos: {np.sum(stroma_candidates):,})")
        axes[1, 1].axis('off')
        
        plt.suptitle(f"Preview - {Path(image_path).name}", fontsize=12)
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Añade clase Stroma a máscaras en zonas de tejido sin anotación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Procesar todo el dataset
  python add_stroma.py /path/to/dataset
  
  # Con parámetros personalizados
  python add_stroma.py /path/to/dataset --blur 7 --threshold 235 --dilate 15
  
  # Preview para ajustar parámetros
  python add_stroma.py /path/to/dataset --preview
  
  # Procesar solo una imagen
  python add_stroma.py /path/to/dataset --name "imagen_001"

Estructura esperada del dataset:
  dataset/
    images/
      imagen_001.ome.tif
      ...
    masks/
      imagen_001__mask_multiclass.ome.tif
      ...
        """
    )
    
    parser.add_argument(
        "dataset_dir",
        help="Directorio del dataset con images/ y masks/"
    )
    parser.add_argument(
        "--output", "-o",
        help="Directorio de salida (default: dataset/masks_with_stroma/)"
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Subdirectorio de imágenes (default: images)"
    )
    parser.add_argument(
        "--masks-dir",
        default="masks",
        help="Subdirectorio de máscaras (default: masks)"
    )
    parser.add_argument(
        "--blur", "-b",
        type=float,
        default=5.0,
        help="Radio del desenfoque gaussiano (default: 5.0)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=240,
        help="Umbral de blancura 0-255 (default: 240, menor = más sensible)"
    )
    parser.add_argument(
        "--min-area", "-a",
        type=int,
        default=1000,
        help="Área mínima de región en píxeles (default: 1000)"
    )
    parser.add_argument(
        "--dilate", "-d",
        type=int,
        default=10,
        help="Radio de dilatación para unir zonas (default: 10)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tamaño de tile para procesamiento (default: 2048)"
    )
    parser.add_argument(
        "--name", "-n",
        help="Procesar solo la imagen que contenga este nombre"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Mostrar preview del thresholding sin guardar"
    )
    parser.add_argument(
        "--preview-level",
        type=int,
        default=2,
        help="Nivel de pirámide para preview (default: 2)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Función principal."""
    args = parse_args()
    
    detector_params = {
        'blur_radius': args.blur,
        'threshold': args.threshold,
        'min_area': args.min_area,
        'dilate_radius': args.dilate,
    }
    
    if args.preview:
        adder = StromaAdder(
            args.dataset_dir,
            output_dir=args.output,
            images_subdir=args.images_dir,
            masks_subdir=args.masks_dir,
            detector_params=detector_params
        )
        
        pairs = adder.find_pairs()
        if args.name:
            pairs = [(i, m) for i, m in pairs if args.name in i.stem]
        
        if not pairs:
            print("No se encontraron pares imagen-máscara")
            sys.exit(1)
        
        img_path, mask_path = pairs[0]
        print(f"Preview de: {img_path.name}")
        
        preview_thresholding(
            str(img_path),
            str(mask_path),
            adder.detector,
            level=args.preview_level
        )
    else:
        print("=" * 60)
        print("Add Stroma to Masks")
        print("=" * 60)
        
        adder = StromaAdder(
            args.dataset_dir,
            output_dir=args.output,
            images_subdir=args.images_dir,
            masks_subdir=args.masks_dir,
            tile_size=args.tile_size,
            detector_params=detector_params
        )
        
        adder.process_all(specific_name=args.name)
        
        print()
        print("=" * 60)
        print("Proceso completado")
        print("=" * 60)


if __name__ == "__main__":
    main()
