import numpy as np
import open3d as o3d
from pathlib import Path

def visualize_pointcloud(npy_path):
    print(f"Cargando nube de puntos desde: {npy_path}")
    
    # 1. Cargar el archivo .npy
    points = np.load(npy_path)
    print(f"Puntos cargados: {points.shape[0]}")
    
    # 2. Crear el objeto PointCloud de Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Opcional: darle un color bonito para la visualización
    pcd.paint_uniform_color([0.2, 0.6, 0.8]) 

    print("Abriendo visualizador de nube de puntos...")
    # 3. Dibujar la nube en una ventana emergente
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Visualizador de Nube de Puntos - Open3D")

if __name__ == '__main__':
    # Usando la ruta que modificaste (PC15k)
    folder_path = Path("ShapeNetCore.v2.PC15k/03001627/train")
    archivos = list(folder_path.glob("*.npy"))
    
    if len(archivos) > 0:
        ejemplo = archivos[0]
        visualize_pointcloud(ejemplo)
    else:
        print(f"No se encontraron archivos en {folder_path}.")
