# Redes Bayesianas para la Detección de Ansiedad a partir de Señales EEG en Jóvenes Adultos

Este repositorio contiene el código fuente, la metodología y los métodos de modelado para la investigación de tesis de maestría titulada **"Redes Bayesianas para la detección de ansiedad a partir de señales de EEG en jóvenes adultos"**.

El proyecto abarca desde la preparación del conjunto de datos electroencefalográficos (EEG), la extracción de características espectrales (potencias por banda, asimetrías hemisféricas y razones entre bandas), la selección de características y la comparación de clasificadores. La inferencia causal requiere supuestos adicionales; las estructuras aprendidas de estos datos observacionales se interpretan como asociaciones compatibles con el modelo, no como efectos causales demostrados.

---

## 📁 Estructura del Repositorio

```text
├── Conjunto de datos/               # Estructura requerida para los datos de entrada (SAM40)
│   └── Data/                        # (Contenido descargable desde Figshare)
│       ├── filtered_data/           # Archivos .mat de EEG filtrados por sujeto
│       ├── scales.xls               # Puntajes de autorreporte SAM (estrés/ansiedad)
│       └── Coordinates.locs         # Coordenadas y etiquetas de los 32 canales EEG
│
├── Código y estructura/             # Scripts ordenados por fase de procesamiento
│   ├── Extracción de características/
│   │   ├── 1_generacion_de_datos.py # Carga .mat, epocamiento (5s, 50% overlap) y etiquetado
│   │   ├── 2_extraccion.py          # Estimación PSD (Welch) log10, asimetrías y ratios (306 feats)
│   │   ├── 3_normalizacion.py       # Z-score respecto al baseline de relajación por sujeto
│   │   ├── 4_graficos_densidad.py   # Gráficos exploratorios de densidad
│   │   └── 5_graficos_boxplot.py    # Boxplots de distribución
│   │
│   ├── Análisis global/              # Modelos clasificadores globales (306 características)
│   │   ├── 1_seleccion_caracteristicas_G.py
│   │   ├── 2_arbol_decision.py
│   │   ├── 3_random_forest.py
│   │   ├── 4_modelo_knn.py
│   │   ├── 5_modelo_svm.py
│   │   ├── 6_modelo_xgboost.py
│   │   └── 9_graficos_metricas_top.py
│   │
│   ├── Análisis por bandas/           # Modelos especializados por banda espectral (α, β, δ, etc.)
│   │   ├── 1_seleccion_caracteristicas_B.py
│   │   ├── 2_modelo_dt.py
│   │   ├── 3_modelo_knn.py
│   │   ├── 4_modelo_svm.py
│   │   └── 5_modelo_xgboost.py
│   │
│   └── Red Bayesiana/                # Modelos de Redes Bayesianas y Descubrimiento Causal
│       ├── Global/
│       │   ├── 10_Comparativa_Bayesiana_MRMR.py # Ejecutor de referencia para la evaluación predictiva
│       │   ├── 1_Red_Bayesiana_Condicional_Gaussiana.py # EEG continuo, clase discreta e inferencia bayesiana
│       │   ├── 2_Red_Discreta.py      # Modelo discreto histórico
│       │   ├── 4_Red_Mixta_Regional.py# Modelo regional histórico
│       │   ├── 5_Ensamble_Conocimiento.py # Ensamble histórico
│       │   ├── 6_Red_TAN.py           # TAN histórico
│       │   └── 8_Red_Naive_Bayes.py   # Naive Bayes gráfico histórico
│       ├── Tareas/
│       │   └── 3_Red_Mixta_Tareas.py  # Modelos específicos (Aritmética, Espejo, Stroop)
│       └── Trials/
│           └── 3_Red_Mixta_Trials.py  # Análisis de dinámica temporal por trial (fatiga)
│
├── Resultados/                      # Carpeta de salida para métricas CSV, DAGs y gráficas (creada automáticamente)
├── environment.yml                  # Configuración de entorno Conda reproducibilidad
├── requirements.txt                 # Dependencias Pip con versiones exactas
└── README.md                        # Guía de reproducción y documentación
```

---

## 🛠️ Requisitos e Instalación

### Opción 1: Entorno Conda (Recomendado)
Para crear e instalar el entorno exacto utilizado en la investigación:

```bash
conda env create -f environment.yml
conda activate bn_anxiety_env
```

### Opción 2: Instalación vía Pip
Si utilizas un entorno virtual existente con **Python 3.10**:

```bash
pip install -r requirements.txt
```

---

## 📊 Conjunto de Datos (SAM 40)

La investigación utiliza la base de datos pública **SAM 40**:
> **SAM 40: Dataset of 40 Subject EEG Recordings to Monitor the Induced-Stress while performing Stroop Color-Word Test, Arithmetic Task and Mirror Image Recognition Task**  
> Disponible en: [Figshare - SAM 40 Dataset](https://figshare.com/articles/dataset/SAM_40_Dataset_of_40_Subject_EEG_Recordings_to_Monitor_the_Induced-Stress_while_performing_Stroop_Color-Word_Test_Arithmetic_Task_and_Mirror_Image_Recognition_Task/14562090/1?file=27956376)

### Instrucciones de colocación:
1. Descarga el archivo de Figshare y descompresiónalo.
2. Coloca la estructura de archivos en la carpeta `Conjunto de datos/Data/` de modo que queden accesibles las siguientes rutas:
   * `Conjunto de datos/Data/filtered_data/*.mat`
   * `Conjunto de datos/Data/scales.xls`
   * `Conjunto de datos/Data/Coordinates.locs`

---

## 🔄 Guía de Reproducción Paso a Paso

Sigue esta secuencia de comandos para reproducir íntegramente los resultados desde cero:

### 1. Preprocesamiento y Extracción de Características
```bash
# 1. Epocamiento y extracción de metadatos/señal raw
python "Código y estructura/Extracción de características/1_generacion_de_datos.py"

# 2. Extracción de potencias PSD log10, asimetrías y ratios (306 características)
python "Código y estructura/Extracción de características/2_extraccion.py"

# 3. Normalización Z-score baseline por sujeto
python "Código y estructura/Extracción de características/3_normalizacion.py"
```

### 2. Selección de Características y Clasificación Tradicional

El selector global separa de forma reproducible 32 sujetos de desarrollo y 8 sujetos de prueba externa (semilla 42); calcula el ranking únicamente con los 32. Los clasificadores tradicionales se ejecutan sobre esos mismos 32 sujetos y comparan Top 10/15/20/30/40 para Fisher, información mutua y mRMR; para importancia DT usan Top 10/15/20, porque el ranking actual solo asigna importancia positiva a 24 variables y los puestos posteriores serían empates en cero. La prueba externa de ocho sujetos se conserva para la evaluación final bayesiana. La validación LOSO tradicional usa un ranking precalculado sobre los 32 y debe interpretarse como comparación de desarrollo, no como estimación libre de sesgo por selección; el ejecutor bayesiano vuelve a calcular las características en cada pliegue.

```bash
# Ranking multicriterio de características (Fisher, MI, mRMR, DT)
python "Código y estructura/Análisis global/1_seleccion_caracteristicas_G.py"

# Entrenamiento de clasificadores globales con validación Leave-One-Subject-Out (LOSO)
python "Código y estructura/Análisis global/2_arbol_decision.py"
python "Código y estructura/Análisis global/3_random_forest.py"
python "Código y estructura/Análisis global/4_modelo_knn.py"
python "Código y estructura/Análisis global/5_modelo_svm.py"
python "Código y estructura/Análisis global/6_modelo_xgboost.py"
```

### 3. Evaluación de Redes Bayesianas

Los resultados predictivos finales se generan con el ejecutor de referencia. Usa el conjunto SAM ≥ 1 acordado, 306 variables candidatas, compara los rankings DT, Fisher, información mutua y mRMR recalculados únicamente en los datos de entrenamiento de cada pliegue, Top global 10/15/20/30/40 para Fisher, información mutua y mRMR; Top 10/15/20 para DT (y Top 10/15/20/30 en bandas) con validación anidada agrupada por sujeto. Compara PC, GES, TAN y Naive Bayes mediante probabilidades posteriores de la red. Reserva los mismos 32 sujetos para desarrollo y 8 para una evaluación externa única.

Ejecuta desde la raíz del repositorio:

```bash
python "Código y estructura/Red Bayesiana/Global/10_Comparativa_Bayesiana_MRMR.py" --analysis global
```

El análisis global se corre por separado del análisis por bandas:

```bash
python "Código y estructura/Red Bayesiana/Global/10_Comparativa_Bayesiana_MRMR.py" --analysis bands
```

También pueden ejecutarse los análisis por tarea y trial con `--analysis tasks` y `--analysis trials`. `--analysis all` ejecuta los alcances global, bandas, tareas y trials y requiere más tiempo. Los análisis por banda usan las variables correspondientes del conjunto de 306 características del artículo. Los scripts anteriores de redes se conservan para consulta y comparación histórica; no deben usarse para reemplazar los resultados del ejecutor de referencia porque algunos clasifican con regresión logística, usan tops distintos o no aplican selección anidada por sujeto.

Los archivos nuevos se guardan en `Resultados/Red Bayesiana/Comparativa_Seleccion_BN_CV/`. La comparación de los cuatro rankings de este ejecutor determina cuál funciona mejor para estas redes bayesianas; no establece cuál selección es mejor para los clasificadores tradicionales.

El ejecutor usa pgmpy, que ya forma parte del entorno fijado. Para controlar el tiempo del algoritmo PC, restringe el orden máximo de los conjuntos condicionantes a dos; esta limitación debe declararse al interpretar la estructura aprendida. No requiere instalar causal-learn: no se ha medido una ventaja de tiempo que justifique añadir otra implementación de PC/GES y conversiones adicionales de grafos.

---

## 📈 Evaluación y Métricas

El ejecutor de referencia usa validación anidada por sujeto para comparar los cuatro métodos de selección y los cinco tamaños globales Top para Fisher, información mutua y mRMR y Top 10/15/20 para DT; en análisis por bandas conserva Top 10/15/20/30. El ranking, la discretización, la estructura y el umbral se ajustan solo con sujetos de entrenamiento. Se usan cuatro pliegues externos y dos internos; la prueba externa se consulta después de fijar una única configuración. Las épocas no se tratan como participantes independientes; además se guardan métricas por sujeto y estabilidad de características/arcos. Las métricas incluyen:
* **Exactitud (Accuracy)**
* **Exactitud balanceada**
* **Sensibilidad (Recall / Sensitivity)**
* **Especificidad (Specificity)**
* **F1-Score, ROC-AUC, PR-AUC y Brier**
* **Intervalos de confianza por remuestreo de sujetos**
* **Frecuencia de selección de características y de arcos**

La normalización existente utiliza el baseline de relajación de cada sujeto. La evaluación asume que ese baseline estará disponible para normalizar las observaciones de un sujeto nuevo; si el caso de uso no dispone de ese baseline, deberá cambiarse también ese protocolo.

---

## 👥 Autores y Afiliación

* **Jovani Gallegos Álvarez**  
  *Universidad Tecnológica de la Mixteca (UTM)*  
  Email: `gaaj010320@gs.utm.mx`
* **Dra. Verónica Rodríguez López**  
  *Universidad Tecnológica de la Mixteca (UTM)*  
  Email: `veromix@gs.utm.mx`
