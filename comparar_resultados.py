import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_old_mean(file_path):
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    nscd_vals = [v for k, v in data.items() if k.endswith("normalized cd")]
    return np.mean(nscd_vals) if nscd_vals else None

def get_new_mean(file_path):
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("mean_nscd", None)

def main():
    categories = ["airplane", "car", "chair"]
    old_dir = Path("Replicability/computed_NSCD")
    new_dir = Path("resultados_icp")
    
    table_data = []
    
    for cat in categories:
        old_file = old_dir / f"shapenet_{cat}.json"
        new_file = new_dir / f"icp_results_{cat}.json"
        
        old_val = get_old_mean(old_file)
        new_val = get_new_mean(new_file)
        
        old_str = f"{old_val:.6f}" if old_val is not None else "N/A"
        new_str = f"{new_val:.6f}" if new_val is not None else "N/A"
        
        table_data.append([cat.capitalize(), old_str, new_str])

    # --- Generación de la imagen de la tabla con Matplotlib ---
    col_labels = ["Categoría", "NSCD Original (ShapeNet)", "NSCD Nuevo (con ICP)"]
    
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear la tabla
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    
    # Estilizar la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)  # Escalar columnas y filas
    
    # Estilizar las celdas de encabezado
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
    
    plt.title("Comparación de NSCD Promedio", fontweight="bold", pad=20)
    
    # Guardar como imagen
    output_image = "comparacion_nscd.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"¡Éxito! La tabla ha sido guardada como imagen en: {output_image}")

if __name__ == "__main__":
    main()
