import os
import numpy as np
import pandas as pd
import configuracion as config
import Preprocesamiento
import Extraccion

def main():
    # Cargar Datos
    tipo_prueba = config.PRUEBA_ARITMETICA
    print(f"Cargando datos para prueba: {tipo_prueba}")
    
    matriz_datos, lista_nombres = Preprocesamiento.cargar_dataset(tipo_prueba=tipo_prueba)
    vector_etiquetas = Preprocesamiento.obtener_etiquetas(tipo_prueba=tipo_prueba)
    
    if matriz_datos.size == 0:
        print("No hay datos cargados. Verificar configuración.")
        return

    # Ajustar dimensiones si es necesario (intersección)
    if matriz_datos.shape[0] != vector_etiquetas.shape[0]:
        minimo = min(matriz_datos.shape[0], vector_etiquetas.shape[0])
        matriz_datos = matriz_datos[:minimo]
        vector_etiquetas = vector_etiquetas[:minimo]
        lista_nombres = lista_nombres[:minimo]
        
    # Dominio del Tiempo
    caract_tiempo = Extraccion.caracteristicas_series_tiempo(matriz_datos)
    
    # Dominio de la Frecuencia
    bandas = [0.5, 4, 8, 13, 30, 45] 
    caract_frec = Extraccion.caracteristicas_bandas_frecuencia(matriz_datos, bandas)
    
    # Complejidad (Hjorth)
    caract_hjorth = Extraccion.caracteristicas_hjorth(matriz_datos)
    
    # Análisis Fractal
    caract_fractal = Extraccion.caracteristicas_fractales(matriz_datos)
    
    # Integración
    matriz_X = np.concatenate([caract_tiempo, caract_frec, caract_hjorth, caract_fractal], axis=1)
    
    # Expansión de etiquetas y metadatos para coincidir con ventanas de tiempo
    n_segs = matriz_datos.shape[1]
    etiquetas_expandidas = np.repeat(vector_etiquetas, n_segs)
    nombres_expandidos = np.repeat(lista_nombres, n_segs)
    
    # Persistencia de resultados
    ruta_base = "Resultados"
    ruta_prueba = os.path.join(ruta_base, tipo_prueba)
    ruta_caract = os.path.join(ruta_prueba, "features")
    
    os.makedirs(ruta_caract, exist_ok=True)
    
    print(f"Guardando en {ruta_prueba}...")
    np.save(os.path.join(ruta_prueba, 'X.npy'), matriz_X)
    np.save(os.path.join(ruta_prueba, 'y.npy'), etiquetas_expandidas)
    np.save(os.path.join(ruta_prueba, 'identifiers.npy'), nombres_expandidos)
    
    # Archivo resumen CSV para inspección
    df_resumen = pd.DataFrame(matriz_X)
    df_resumen.insert(0, "Identificador", nombres_expandidos)
    df_resumen.insert(1, "Etiqueta", etiquetas_expandidas)
    df_resumen.to_csv(os.path.join(ruta_prueba, 'dataset_completo.csv'), index=False)
    
    # Guardado modular
    np.save(os.path.join(ruta_caract, 'time.npy'), caract_tiempo)
    np.save(os.path.join(ruta_caract, 'frequency.npy'), caract_frec)
    np.save(os.path.join(ruta_caract, 'hjorth.npy'), caract_hjorth)
    np.save(os.path.join(ruta_caract, 'fractal.npy'), caract_fractal)
    
    print("Ejecución finalizada.")

if __name__ == "__main__":
    main()