# Redes Bayesianas para la Detección de Ansiedad a partir de Señales EEG en Jóvenes Adultos

Este repositorio contiene el código fuente, la metodología y los métodos de modelado para la investigación de tesis de maestría titulada **"Redes Bayesianas para la detección de ansiedad a partir de señales de EEG en jóvenes adultos"**.

El proyecto abarca desde la preparación del conjunto de datos electroencefalográficos (EEG), la extracción de características espectrales (potencias por banda, asimetrías hemisféricas y ratios inter-banda), la selección multicriterio de características y la construcción de clasificadores tradicionales, hasta el modelado estructural e inferencia causal mediante **Redes Bayesianas** (Globales, Regionales, por Tareas y Ensamble de Conocimiento).

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
│       │   ├── 1_Red_Continua.py      # Red Gaussiana Lineal (PC / GES)
│       │   ├── 2_Red_Discreta.py      # Red Bayesiana Discreta
│       │   ├── 3_Red_Mixta.py         # Red Mixta Continuas-Discreta (Manta de Markov + RegLog)
│       │   ├── 4_Red_Mixta_Regional.py# Agrupación anatómica en 5 regiones corticales
│       │   ├── 5_Red_Ensamble_Conocimiento.py # Ensamble (Ponderado, DAG Priors, Stacking)
│       │   ├── 6_Red_TAN.py           # Tree-Augmented Naive Bayes
│       │   └── 8_Red_Naive_Bayes.py   # Naive Bayes gráfico
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

### 3. Construcción y Evaluación de Redes Bayesianas
```bash
# Red Bayesiana Gaussiana Lineal Continua
python "Código y estructura/Red Bayesiana/Global/1_Red_Continua.py"

# Red Bayesiana Mixta (Descubrimiento causal PC/GES + Manta de Markov + RegLog)
python "Código y estructura/Red Bayesiana/Global/3_Red_Mixta.py"

# Red Bayesiana Mixta Regional (5 regiones corticales: Frontal, Central, Temporal, Parietal, Occipital)
python "Código y estructura/Red Bayesiana/Global/4_Red_Mixta_Regional.py"

# Red Bayesiana por Tareas específicas (Aritmética, Espejo, Stroop)
python "Código y estructura/Red Bayesiana/Tareas/3_Red_Mixta_Tareas.py"

# Red Bayesiana por Trials (Evolución temporal y fatiga)
python "Código y estructura/Red Bayesiana/Trials/3_Red_Mixta_Trials.py"

# Red de Ensamble de Conocimiento (Fusión Ponderada, DAG Priors y Stacking)
python "Código y estructura/Red Bayesiana/Global/5_Red_Ensamble_Conocimiento.py"
```

---

## 📈 Evaluación y Métricas

Todos los modelos se evalúan mediante **Validación Cruzada Dejando un Sujeto Fuera (Leave-One-Subject-Out - LOSO)** para prevenir fugas de información entre sujetos y evaluar la capacidad de generalización inter-sujeto real. Las métricas reportadas incluyen:
* **Exactitud (Accuracy)**
* **Sensibilidad (Recall / Sensitivity)**
* **Especificidad (Specificity)**
* **F1-Score**
* **Matrices de Confusión acumuladas** y visualización de grafos causales (DAGs).

---

## 👥 Autores y Afiliación

* **Jovani Gallegos Álvarez**  
  *Universidad Tecnológica de la Mixteca (UTM)*  
  Email: `gaaj010320@gs.utm.mx`
* **Dra. Verónica Rodríguez López**  
  *Universidad Tecnológica de la Mixteca (UTM)*  
  Email: `veromix@gs.utm.mx`
