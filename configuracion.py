# Configuración global del proyecto

# Rutas de directorios
DIR_BASE = 'Conjunto de datos/Data/'
DIR_DATOS_CRUDOS = DIR_BASE + 'raw_data'
DIR_DATOS_FILTRADOS = DIR_BASE + 'filtered_data'
DIR_DATOS_FILTRADOS_ICA = DIR_BASE + 'artifact_removal' 
RUTA_ETIQUETAS = DIR_BASE + 'scales.xls'

# Diccionario de renombrado de columnas para el archivo Excel
COLUMNAS_A_RENOMBRAR = {
    'Subject No.': 'n_sujeto',
    'Trial_1': 't1_mate',
    'Unnamed: 2': 't1_espejo',
    'Unnamed: 3': 't1_stroop',
    'Trial_2': 't2_mate',
    'Unnamed: 5': 't2_espejo',
    'Unnamed: 6': 't2_stroop',
    'Trial_3': 't3_mate',
    'Unnamed: 8': 't3_espejo',
    'Unnamed: 9': 't3_stroop'
}

# Tipos de datos soportados
TIPOS_DE_DATOS = ["crudos", "filtrados_wt", "filtrados_ica"]

# Identificadores de pruebas experimentales
PRUEBA_ARITMETICA = "Arithmetic"
PRUEBA_ESPEJO = "Mirror"
PRUEBA_STROOP = "Stroop"
TIPOS_DE_PRUEBA = [PRUEBA_ARITMETICA, PRUEBA_ESPEJO, PRUEBA_STROOP]

# Asociación de columnas de etiquetas por tipo de prueba
COLUMNAS_TIPO_PRUEBA = {
    PRUEBA_ARITMETICA: ['t1_mate', 't2_mate', 't3_mate'],
    PRUEBA_ESPEJO: ['t1_espejo', 't2_espejo', 't3_espejo'],
    PRUEBA_STROOP: ['t1_stroop', 't2_stroop', 't3_stroop']
}

N_CLASES = 2
FRECUENCIA_MUESTREO = 128
