import numpy as np
import pandas as pd
import os
import sys

def convertir_npy_a_csv(archivo_npy):
    try:
        # Cargar datos
        data = np.load(archivo_npy)
        print(f"Abriendo {archivo_npy}...")
        print(f"Forma original: {data.shape}")
        
        # Si es 3D o más (ej. trials x tiempo x canales), aplanar para CSV 2D
        if data.ndim > 2:
            print("Matriz multidimensional detectada. Aplanando a 2D para exportación...")
            # Opción simple: aplanar todo excepto la primera dimensión (samples/trials)
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data
            
        # Crear nombre de salida
        archivo_csv = archivo_npy.replace('.npy', '.csv')
        
        # Guardar (usando pandas es rápido para CSVs grandes)
        df = pd.DataFrame(data_2d)
        print(f"Guardando en {archivo_csv}...")
        df.to_csv(archivo_csv, index=False)
        print("¡Listo!")
        
    except Exception as e:
        print(f"Error procesando {archivo_npy}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convertir_npy.py <archivo.npy>")
    else:
        convertir_npy_a_csv(sys.argv[1])