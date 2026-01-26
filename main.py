import os
import numpy as np
import pandas as pd
import configuracion as config
import Preprocesamiento
import Extraccion

def main():
    # 1. Cargar Datos
    prueba = config.PRUEBA_ARITMETICA
    print(f"Cargando datos para prueba: {prueba}")
    
    # Modificación: Recibir también los nombres de archivos
    datos, nombres = Preprocesamiento.cargar_dataset(tipo_prueba=prueba)
    etiquetas = Preprocesamiento.obtener_etiquetas(tipo_prueba=prueba)
    
    if datos.size == 0:
        print("No se cargaron datos. Verifica las rutas en configuracion.py")
        return

    print(f"Datos cargados: {datos.shape}") # (n_trials, n_secs, n_channels, n_samples)
    print(f"Etiquetas cargadas: {etiquetas.shape}")
    
    # Verificar alineación
    if datos.shape[0] != etiquetas.shape[0]:
        print(f"ADVERTENCIA: Desajuste en número de muestras. Datos: {datos.shape[0]}, Etiquetas: {etiquetas.shape[0]}")
        # Ajustar al mínimo común
        min_len = min(datos.shape[0], etiquetas.shape[0])
        datos = datos[:min_len]
        etiquetas = etiquetas[:min_len]
        nombres = nombres[:min_len]
        
    
    # 2. Extracción de Características
    print("Extrayendo características...")
    
    # Características de Tiempo
    print("- Series de tiempo (Varianza, RMS, PTP)")
    feat_time = Extraccion.caracteristicas_series_tiempo(datos)
    print(f"  Shape: {feat_time.shape}")
    
    # Características de Frecuencia
    print("- Bandas de frecuencia")
    # Bandas estándar: Delta, Theta, Alpha, Beta, Gamma (baja)
    bandas = [0.5, 4, 8, 13, 30, 45] 
    feat_freq = Extraccion.caracteristicas_bandas_frecuencia(datos, bandas)
    print(f"  Shape: {feat_freq.shape}")
    
    # Características de Hjorth
    print("- Hjorth (Movilidad, Complejidad)")
    feat_hjorth = Extraccion.caracteristicas_hjorth(datos)
    print(f"  Shape: {feat_hjorth.shape}")
    
    # Características Fractales
    print("- Fractales (Higuchi, Katz)")
    feat_fractal = Extraccion.caracteristicas_fractales(datos)
    print(f"  Shape: {feat_fractal.shape}")
    
    # 3. Concatenar todas las características
    X = np.concatenate([feat_time, feat_freq, feat_hjorth, feat_fractal], axis=1)
    
    # Extraccion.py devuelve (n_trials * n_secs, n_features).
    # Las etiquetas originalson (n_trials,).
    # Necesitamos expandir 'y' para que coincida con X.
    n_secs = datos.shape[1]
    y_expanded = np.repeat(etiquetas, n_secs)
    
    # Expandir nombres para que coincidan con X
    nombres_expanded = np.repeat(nombres, n_secs)
    
    print(f"Matriz de características final X: {X.shape}")
    print(f"Vector de etiquetas final y: {y_expanded.shape}")
    print(f"Vector de identificadores: {nombres_expanded.shape}")
    
    # 4. Guardar resultados
    base_dir = "Resultados"
    prueba_dir = os.path.join(base_dir, prueba)
    features_dir = os.path.join(prueba_dir, "features")
    
    # Crear directorios si no existen
    os.makedirs(features_dir, exist_ok=True)
    
    print(f"Guardando resultados en {prueba_dir}...")
    np.save(os.path.join(prueba_dir, 'X.npy'), X)
    np.save(os.path.join(prueba_dir, 'y.npy'), y_expanded)
    np.save(os.path.join(prueba_dir, 'identifiers.npy'), nombres_expanded)
    
    # Guardar un CSV con Identificador, Etiqueta y Características (opcional, útil para debugging)
    df_resumen = pd.DataFrame(X)
    df_resumen.insert(0, "Identificador", nombres_expanded)
    df_resumen.insert(1, "Etiqueta", y_expanded)
    df_resumen.to_csv(os.path.join(prueba_dir, 'dataset_completo.csv'), index=False)
    
    # Guardar características individuales
    print(f"Guardando características individuales en {features_dir}...")
    np.save(os.path.join(features_dir, 'time.npy'), feat_time)
    np.save(os.path.join(features_dir, 'frequency.npy'), feat_freq)
    np.save(os.path.join(features_dir, 'hjorth.npy'), feat_hjorth)
    np.save(os.path.join(features_dir, 'fractal.npy'), feat_fractal)
    
    print("Proceso completado.")

if __name__ == "__main__":
    main()
