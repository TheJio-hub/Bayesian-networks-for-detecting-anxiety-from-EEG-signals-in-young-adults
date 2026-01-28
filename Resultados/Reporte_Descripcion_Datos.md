# Reporte Técnico: Descripción de Datos y Estrategia de Análisis

## 1. Fuente de Datos y Preprocesamiento

### Dimensión de Entrada
Partimos de estructuras de datos EEG preprocesadas (filtradas y limpias de artefactos) con la siguiente organización n-dimensional:
- **Tensor Original:** `(N_Trials, N_Segundos, N_Canales, N_Muestras)`
- **Resolución:**
  - **Canales:** 32 electrodos (posicionados según estándar 10-20).
  - **Frecuencia de Muestreo:** 250 Hz (o la definida en `configuracion.py`).
  - **Ventana temporal:** 1 segundo por segmento (Epochs de 1s sin solapamiento).

### Correspondencia Trial-Estímulo
La estrategia de etiquetado se define por la naturaleza de la tarea experimental:

| Clase | Etiqueta | Tarea | Selección Temporal | Descripción |
| :--- | :---: | :--- | :--- | :--- |
| **Relajación** | `0` | Baseline / Reposo | **Primeros 25 segundos** | Se recorta estricta y únicamente el inicio de la prueba para garantizar un estado basal limpio, libre de aburrimiento o divagación mental tardía. |
| **Ansiedad** | `1` | Aritmética Mental | **Todo el trial** | Se utilizan todos los segmentos disponibles durante la resolución de problemas matemáticos bajo presión, capturando la respuesta de estrés completa. |

---

## 2. Construcción del Espacio de Características (Matriz X)

Cada segmento de 1 segundo se trata como una instancia independiente para el clasificador.

### Dimensiones Finales
- **Total de Muestras (Instancias):** ~4,425 epochs
  - Clase 0 (Relax): 3,000 epochs (aprox. 120 trials × 25s).
  - Clase 1 (Ansiedad): 1,425 epochs (variable según duración de trials de estrés).
- **Total de Características (Columnas):** 384 variables
  - Fórmula: `32 canales × 12 métricas = 384 features`.

### Desglose de Características
Por cada canal se extraen 4 grupos de métricas:

1.  **Dominio del Tiempo (3):** Varianza, RMS (Root Mean Square), Amplitud Pico-a-Pico.
2.  **Dominio de Frecuencia (5):** Potencia espectral en bandas Delta, Theta, Alpha, Beta, Gamma.
3.  **Parámetros de Hjorth (2):** Movilidad, Complejidad.
4.  **Dimensión Fractal (2):** Higuchi, Katz.

---

## 3. Resumen de Resultados de Separabilidad

### Análisis Discriminante (Ranking Fisher)
Se evaluó individualmente el poder predictivo de cada característica. Los resultados indican que los marcadores más fuertes no son bandas de frecuencia simples, sino medidas de complejidad de la señal.
- **Top Feature:** `Hjorth Mobility` en canales fronto-centrales (Fp1, F8, Cz, Fz).
- **Interpretación:** La "forma" y complejidad de la onda cambia más drásticamente entre relajación y ansiedad que la potencia pura de una banda específica.

### Análisis de Proyección (PCA / t-SNE)
Las proyecciones bidimensionales confirman agrupamientos observables pero con superposición, lo sugerente de que:
- **Análisis Global:** Usar proyecciones de todos los canales mezcla ruido.
- **Análisis Regional:** Las proyecciones enfocadas en **regiones individuales** (ej. Frontal) y **métricas específicas** (ej. Hjorth o Fractal) muestran una separación mucho más limpia entre las nubes de puntos roja (Ansiedad) y azul (Relax).

---

## 4. Archivos Generados para Revisión

1.  **`Resultados/Evaluacion_Caracteristicas.csv`**: Ranking detallado con Score Fisher e Información Mutua de las 384 variables.
2.  **`Resultados/Top20_Features_Fisher.png`**: Gráfico de barras de las 20 mejores características.
3.  **`Resultados/Proyecciones_Canales_Detalle/`**: Carpeta con 96 imágenes de PCA/t-SNE desglosadas por canal y métrica para validación topológica.
