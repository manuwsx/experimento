import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from pathlib import Path

def householder_reflection(points):
    # Definición de la matriz de Householder para el plano YZ (x=0)
    v = np.array([1, 0, 0])
    H = np.eye(3) - 2 * np.outer(v, v)
    return points @ H.T

def compute_diagonal(points):
    # Computa la diagonal del axis-aligned bounding box
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    return np.sqrt(dx**2 + dy**2 + dz**2)

def chamfer_distance(pc1, pc2):
    # Cálculo bidireccional usando KDTree
    tree1 = KDTree(pc2)
    dist1 = tree1.query(pc1)[0]
    
    tree2 = KDTree(pc1)
    dist2 = tree2.query(pc2)[0]
    
    return np.mean(dist1) + np.mean(dist2)

def evaluate_nscd_with_icp(original_points, threshold=0.05):
    """
    Toma la nube de puntos original, aplica la transformación de Householder,
    utiliza ICP para alinearla y finalmente calcula el NSCD.
    """
    # 1. Aplicar la transformación de Householder inicial respecto a x=0
    reflected_points = householder_reflection(original_points)
    
    # 2. Conversión a la estructura PointCloud de Open3D
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(original_points)
    
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(reflected_points)
    
    # 3. Aplicar Iterative Closest Point (ICP) para corregir desviación ligera
    initial_transform = np.eye(4)
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_ref, pcd_orig, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # Aplicar la matriz rígida encontrada a la nube reflejada
    pcd_ref.transform(icp_result.transformation)
    aligned_reflected_points = np.asarray(pcd_ref.points)
    
    # 4. Obtener la distancia de Chamfer (CD)
    cd = chamfer_distance(original_points, aligned_reflected_points)
    
    # 5. Normalizar respecto al cuadrado de la diagonal (NSCD)
    diagonal = compute_diagonal(original_points)
    nscd = cd / (diagonal**2)
    
    return nscd, icp_result.transformation

def process_categories(base_dir, output_dir):
    """Evalúa exactamente 1000 muestras por cada categoría de ShapeNet."""
    categories = {
        "Airplane": "02691156",
        "Car": "02958343",
        "Chair": "03001627"
    }
    
    base_path = Path(base_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name, synset_id in categories.items():
        class_path = base_path / synset_id / "train"
        
        # Obtener y limitar a exactamente 1000 muestras
        files = list(class_path.rglob("*.npy"))[:1000]
        
        if not files:
            print(f"No se encontraron archivos para {class_name} en {class_path}")
            continue
            
        print(f"\nProcesando {class_name} ({len(files)} muestras)...")
        
        class_results = {}
        nscd_values = []
        
        for idx, f in enumerate(files, 1):
            try:
                points = np.load(f)
                
                # Desempaquetado correcto de la tupla retornada
                nscd, _ = evaluate_nscd_with_icp(points)
                
                nscd_values.append(nscd)
                class_results[f.name] = nscd
                
                if idx % 100 == 0:
                    print(f"  Analizados {idx}/1000...")
            except Exception as e:
                print(f"  Error en {f.name}: {e}")
                
        # Calcular el promedio general de la categoría
        mean_nscd = np.mean(nscd_values) if nscd_values else 0
        print(f"==> Promedio NSCD (con ICP) para {class_name}: {mean_nscd:.6f}")
        
        # Guardar los resultados individuales en JSON
        json_path = Path(output_dir) / f"icp_results_{class_name.lower()}.json"
        with open(json_path, 'w') as file:
            json.dump({
                "mean_nscd": mean_nscd,
                "samples": class_results
            }, file, indent=4)

if __name__ == '__main__':
    # Ruta base a las categorías submuestreadas a 2048 puntos
    dataset_2048_dir = "ShapeNetCore.v2.PC2048"
    
    # Directorio para guardar los resultados
    resultados_dir = "resultados_icp"
    
    process_categories(dataset_2048_dir, resultados_dir)