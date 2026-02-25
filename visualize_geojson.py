#!/usr/bin/env python3
"""
Script para visualizar contornos de archivos GeoJSON de QuPath.
- Muestra polígonos coloreados por clase
- Detecta y lista polígonos con múltiples listas de puntos (MultiPolygon o con huecos)
- Navega entre archivos al cerrar cada ventana
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import numpy as np


def generate_class_colors(class_names: set[str]) -> dict[str, tuple[float, float, float]]:
    """Genera colores únicos para cada clase usando un colormap."""
    cmap = plt.cm.get_cmap('tab20', len(class_names))
    colors = {}
    for i, name in enumerate(sorted(class_names)):
        rgba = cmap(i)
        colors[name] = (rgba[0], rgba[1], rgba[2])
    return colors


def adjust_color_brightness(
    color: tuple[float, float, float], 
    factor: float
) -> tuple[float, float, float]:
    """
    Ajusta el brillo de un color.
    
    Args:
        color: Tupla RGB con valores 0-1
        factor: Factor de brillo (< 1 = más oscuro, > 1 = más claro)
    
    Returns:
        Color ajustado (RGB 0-1)
    """
    r, g, b = color
    r = max(0.0, min(1.0, r * factor))
    g = max(0.0, min(1.0, g * factor))
    b = max(0.0, min(1.0, b * factor))
    return (r, g, b)


def extract_all_rings_from_feature(
    feature: dict[str, Any]
) -> list[tuple[np.ndarray, int]]:
    """
    Extrae TODAS las listas/anillos de una feature para visualización multi-lista.
    
    Returns:
        Lista de tuplas (array_de_puntos, índice_de_lista)
    """
    geometry = feature.get('geometry', {})
    geom_type = geometry.get('type', '')
    coords = geometry.get('coordinates', [])
    
    rings: list[tuple[np.ndarray, int]] = []
    
    if geom_type == 'Polygon':
        for i, ring in enumerate(coords):
            if ring:
                rings.append((np.array(ring), i))
    elif geom_type == 'MultiPolygon':
        for i, poly in enumerate(coords):
            if poly and len(poly) > 0:
                rings.append((np.array(poly[0]), i))
    
    return rings


def get_classification_name(feature: dict[str, Any]) -> str:
    """Extrae el nombre de clasificación de una feature."""
    props = feature.get('properties', {})
    classification = props.get('classification', {})
    return classification.get('name', 'Sin clasificación')


def analyze_geometry(feature: dict[str, Any]) -> tuple[str, list[list[list[float]]], int]:
    """
    Analiza la geometría de una feature.
    
    Returns:
        Tuple con (tipo_geometría, lista_de_anillos, num_listas)
        - Para Polygon: num_listas es el número de anillos (1 = solo exterior, >1 = con huecos)
        - Para MultiPolygon: num_listas es el número total de polígonos
    """
    geometry = feature.get('geometry', {})
    geom_type = geometry.get('type', '')
    coords = geometry.get('coordinates', [])
    
    if geom_type == 'Polygon':
        return geom_type, coords, len(coords)
    elif geom_type == 'MultiPolygon':
        return geom_type, coords, len(coords)
    else:
        return geom_type, [], 0


def extract_polygons_for_plot(feature: dict[str, Any]) -> list[np.ndarray]:
    """
    Extrae todos los polígonos de una feature para dibujar.
    Solo devuelve los anillos exteriores (ignora huecos para simplificar visualización).
    """
    geometry = feature.get('geometry', {})
    geom_type = geometry.get('type', '')
    coords = geometry.get('coordinates', [])
    
    polygons = []
    
    if geom_type == 'Polygon':
        if coords and len(coords) > 0:
            polygons.append(np.array(coords[0]))
    elif geom_type == 'MultiPolygon':
        for poly in coords:
            if poly and len(poly) > 0:
                polygons.append(np.array(poly[0]))
    
    return polygons


def load_geojson(filepath: str) -> list[dict[str, Any]]:
    """Carga un archivo GeoJSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_geojson(
    filepath: str, 
    class_colors: dict[str, tuple[float, float, float]],
    only_multi: bool = False
) -> bool:
    """
    Visualiza un archivo GeoJSON y lista los polígonos con múltiples listas.
    
    Args:
        filepath: Ruta al archivo GeoJSON
        class_colors: Diccionario de colores por clase
        only_multi: Si True, solo muestra polígonos con más de una lista
                    y usa colores por instancia (no por clase)
        
    Returns:
        True si hay algo que mostrar, False si no hay nada (cuando only_multi=True)
    """
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Archivo: {filename}")
    print(f"{'='*60}")
    
    features = load_geojson(filepath)
    
    multi_list_features: list[dict[str, Any]] = []
    polygons_by_class: dict[str, list[np.ndarray]] = {}
    
    for feature in features:
        class_name = get_classification_name(feature)
        geom_type, coords, num_lists = analyze_geometry(feature)
        
        is_multi = num_lists > 1
        
        if is_multi:
            multi_list_features.append({
                'id': feature.get('id', 'N/A'),
                'class': class_name,
                'type': geom_type,
                'num_lists': num_lists,
                'feature': feature
            })
        
        if only_multi and not is_multi:
            continue
        
        if not only_multi:
            if class_name not in polygons_by_class:
                polygons_by_class[class_name] = []
            polys = extract_polygons_for_plot(feature)
            polygons_by_class[class_name].extend(polys)
    
    if multi_list_features:
        print(f"\n>>> POLÍGONOS CON MÁS DE UNA LISTA DE PUNTOS: {len(multi_list_features)}")
        print("-" * 60)
        for item in multi_list_features:
            print(f"  - ID: {item['id'][:20]}...")
            print(f"    Clase: {item['class']}")
            print(f"    Tipo: {item['type']}")
            print(f"    Número de listas: {item['num_lists']}")
            print()
    else:
        print("\n>>> No hay polígonos con más de una lista de puntos.")
    
    if only_multi and not multi_list_features:
        print("    (Saltando visualización - no hay multi-lista en este archivo)")
        return False
    
    mode_str = " [SOLO MULTI-LISTA]" if only_multi else ""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    legend_patches = []
    
    if only_multi:
        instance_cmap = plt.cm.get_cmap('hsv', len(multi_list_features) + 1)
        total_rings = 0
        
        for idx, item in enumerate(multi_list_features):
            feature = item['feature']
            base_rgba = instance_cmap(idx)
            base_color = (base_rgba[0], base_rgba[1], base_rgba[2])
            
            rings = extract_all_rings_from_feature(feature)
            num_rings = len(rings)
            total_rings += num_rings
            
            for ring_array, ring_idx in rings:
                if num_rings > 1:
                    brightness = 1.0 - (ring_idx * 0.3)
                    brightness = max(0.4, brightness)
                else:
                    brightness = 1.0
                
                ring_color = adjust_color_brightness(base_color, brightness)
                
                collection = PolyCollection(
                    [ring_array],
                    facecolors=[(*ring_color, 0.5)],
                    edgecolors=[(*ring_color, 1.0)],
                    linewidths=1.0
                )
                ax.add_collection(collection)
            
            patch = mpatches.Patch(
                color=base_color, 
                label=f'Inst {idx+1}: {item["class"][:15]} ({num_rings} listas)'
            )
            legend_patches.append(patch)
        
        ax.set_title(
            f'{filename}{mode_str}\n'
            f'(Multi-lista: {len(multi_list_features)} instancias, '
            f'Total anillos: {total_rings})'
        )
        print(f"\nInstancias multi-lista: {len(multi_list_features)}")
    else:
        total_polys = sum(len(p) for p in polygons_by_class.values())
        if total_polys == 0:
            print("    (No hay polígonos para mostrar)")
            plt.close(fig)
            return False
        
        print(f"\nClases encontradas: {list(polygons_by_class.keys())}")
        
        for class_name, polys in polygons_by_class.items():
            if not polys:
                continue
            
            color = class_colors.get(class_name, (0.5, 0.5, 0.5))
            
            collection = PolyCollection(
                polys,
                facecolors=[(*color, 0.4)],
                edgecolors=[(*color, 1.0)],
                linewidths=0.5
            )
            ax.add_collection(collection)
            
            patch = mpatches.Patch(color=color, label=f'{class_name} ({len(polys)})')
            legend_patches.append(patch)
        
        ax.set_title(
            f'{filename}{mode_str}\n'
            f'(Total features: {len(features)}, Multi-lista: {len(multi_list_features)}, '
            f'Mostrando: {total_polys} polígonos)'
        )
    
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    ax.legend(
        handles=legend_patches, 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1),
        fontsize=8
    )
    
    plt.tight_layout()
    plt.show()
    return True


def collect_all_classes_and_counts(
    geojson_files: list[str]
) -> tuple[set[str], dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """
    Recolecta todas las clases únicas, cuenta etiquetas por archivo y por clase.
    
    Returns:
        Tuple con:
        - conjunto_de_clases
        - diccionario_filepath->num_etiquetas
        - diccionario_clase->num_instancias_totales
        - diccionario_filepath->diccionario_clase->num_instancias (desglose por archivo)
    """
    all_classes: set[str] = set()
    file_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    file_class_counts: dict[str, dict[str, int]] = {}
    
    for filepath in geojson_files:
        try:
            features = load_geojson(filepath)
            file_counts[filepath] = len(features)
            file_class_counts[filepath] = {}
            
            for feature in features:
                class_name = get_classification_name(feature)
                all_classes.add(class_name)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                file_class_counts[filepath][class_name] = file_class_counts[filepath].get(class_name, 0) + 1
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")
            file_counts[filepath] = 0
            file_class_counts[filepath] = {}
    
    return all_classes, file_counts, class_counts, file_class_counts


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Visualiza contornos de archivos GeoJSON de QuPath'
    )
    parser.add_argument(
        '--only-multi',
        action='store_true',
        help='Solo muestra etiquetas con más de una lista de puntos (MultiPolygon o con huecos)'
    )
    parser.add_argument(
        '--path',
        type=str,
        default="/home/abel/phd/qupath-annotations/buenas",
        help='Ruta a la carpeta con los archivos GeoJSON'
    )
    return parser.parse_args()


def main() -> None:
    """Función principal."""
    args = parse_args()
    
    data_path = args.path
    only_multi = args.only_multi
    
    geojson_files: list[str] = []
    dir_path = Path(data_path)

    if dir_path.exists():
        files = sorted(dir_path.glob('*.geojson'))
        geojson_files.extend([str(f) for f in files])

    if not geojson_files:
        print(f"No se encontraron archivos GeoJSON en la carpeta '{data_path}'")
        return

    print(f"Encontrados {len(geojson_files)} archivos GeoJSON")
    if only_multi:
        print(">>> MODO: Solo etiquetas con más de una lista (--only-multi)")
    print("Recolectando clases y contando etiquetas de todos los archivos...")

    all_classes, file_counts, class_counts, file_class_counts = collect_all_classes_and_counts(geojson_files)
    class_colors = generate_class_colors(all_classes)

    print(f"\n{'='*60}")
    print("RESUMEN DE ETIQUETAS POR IMAGEN (desglose por clase)")
    print(f"{'='*60}")
    total_labels = 0
    for filepath in geojson_files:
        count = file_counts.get(filepath, 0)
        total_labels += count
        filename = os.path.basename(filepath)
        print(f"\n  {filename}: {count} etiquetas")
        
        # Mostrar desglose por clase para este archivo
        classes_in_file = file_class_counts.get(filepath, {})
        for class_name in sorted(classes_in_file.keys()):
            class_count = classes_in_file[class_name]
            print(f"      - {class_name}: {class_count}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_labels} etiquetas en {len(geojson_files)} imágenes")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"CLASES ENCONTRADAS ({len(all_classes)}) - SUMA TOTAL TODAS LAS IMÁGENES")
    print(f"{'='*60}")
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"  {class_name}: {count}")
    print(f"{'-'*60}")
    print(f"  TOTAL: {sum(class_counts.values())} etiquetas")
    print(f"{'='*60}")
    print(f"\n{'#'*60}")
    print("Iniciando visualización. Cierra cada ventana para ver el siguiente archivo.")
    if only_multi:
        print("(Archivos sin multi-lista serán saltados automáticamente)")
    print(f"{'#'*60}")
    
    shown_count = 0
    for i, filepath in enumerate(geojson_files):
        num_labels = file_counts.get(filepath, 0)
        print(f"\n[{i+1}/{len(geojson_files)}] Mostrando: {os.path.basename(filepath)}")
        print(f"    >>> Esta imagen tiene {num_labels} etiquetas <<<")
        try:
            was_shown = visualize_geojson(filepath, class_colors, only_multi=only_multi)
            if was_shown:
                shown_count += 1
        except Exception as e:
            print(f"Error visualizando {filepath}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Visualización completada.")
    if only_multi:
        print(f"Se mostraron {shown_count} de {len(geojson_files)} archivos (con multi-lista)")
    print("="*60)


if __name__ == '__main__':
    main()
