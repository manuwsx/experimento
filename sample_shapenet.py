import os
from pathlib import Path
import numpy as np
import open3d as o3d

def farthest_point_sampling(points, n_samples=2048):
    """Submuestrea la nube manteniendo la distribución geométrica."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.farthest_point_down_sample(n_samples)
    return np.asarray(downpcd.points)

def sample_shapenet_class(input_dir, output_dir, n_samples=2048):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Usamos rglob por si los archivos están dentro de subcarpetas
    files = list(input_path.rglob("*.npy"))
    total = len(files)
    
    print(f"Ruta: {input_path.resolve()}")
    print(f"Archivos encontrados: {total}")
    
    if total == 0:
        print("No se encontraron archivos .npy en el directorio.")
        return

    print("Iniciando submuestreo secuencial...")
    
    for idx, file_path in enumerate(files, 1):
        try:
            points = np.load(file_path)
            
            # Submuestrear solo si excede la cantidad deseada
            if points.shape[0] > n_samples:
                sampled_points = farthest_point_sampling(points, n_samples)
            else:
                sampled_points = points
                
            # Guardar plano en la carpeta destino con el nombre del archivo
            dest_file = output_path / file_path.name
            np.save(dest_file, sampled_points)
            
            if idx % 50 == 0 or idx == total:
                print(f"[{idx}/{total}] Procesado: {file_path.name} ({points.shape[0]} -> {sampled_points.shape[0]})")
                
        except Exception as e:
            print(f"Error procesando {file_path.name}: {e}")

if __name__ == '__main__':
    # Ajusta las rutas según donde te encontró los archivos
    input_category_path = "ShapeNetCore.v2.PC15k/03001627/train"
    output_category_path = "ShapeNetCore.v2.PC2048/03001627/train"

    sample_shapenet_class(input_category_path, output_category_path)