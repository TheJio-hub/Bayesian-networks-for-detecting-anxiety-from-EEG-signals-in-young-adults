import mne_features.univariate as mne_f
import numpy as np

def caracteristicas_series_tiempo(matriz_datos):
    '''
    Calcula Varianza, RMS y Amplitud Pico-a-Pico.
    Args:
        matriz_datos (ndarray): Datos EEG (Trials, Segs, Canales, Muestras).
    Returns:
        ndarray: Matriz de características aplanada (Muestras, Features).
    '''
    n_trials, n_segs, n_canales, _ = matriz_datos.shape
    num_feat_canal = 3

    matriz_caract = np.empty([n_trials, n_segs, n_canales * num_feat_canal])
    
    for i, trial in enumerate(matriz_datos):
        for j, segmento in enumerate(trial):
            varianza = mne_f.compute_variance(segmento)
            rms = mne_f.compute_rms(segmento)
            ptp = mne_f.compute_ptp_amp(segmento)
            matriz_caract[i][j] = np.concatenate([varianza, rms, ptp])
    
    return matriz_caract.reshape([n_trials*n_segs, n_canales*num_feat_canal])


def caracteristicas_bandas_frecuencia(matriz_datos, limites_bandas):
    '''
    Calcula Densidad Espectral de Potencia (PSD) en bandas específicas.
    '''
    n_trials, n_segs, n_canales, frec_muestreo = matriz_datos.shape
    num_bandas = len(limites_bandas)-1

    matriz_caract = np.empty([n_trials, n_segs, n_canales * num_bandas])
    
    for i, trial in enumerate(matriz_datos):
        for j, segmento in enumerate(trial):
            psd = mne_f.compute_pow_freq_bands(
                frec_muestreo, segmento, freq_bands=limites_bandas)
            matriz_caract[i][j] = psd
            
    return matriz_caract.reshape([n_trials*n_segs, n_canales*num_bandas])

def caracteristicas_hjorth(matriz_datos):
    '''Calcula Movilidad y Complejidad de Hjorth.'''
    n_trials, n_segs, n_canales, _ = matriz_datos.shape
    num_feat = 2

    matriz_caract = np.empty([n_trials, n_segs, n_canales * num_feat])
    
    for i, trial in enumerate(matriz_datos):
        for j, segmento in enumerate(trial):
            movilidad = mne_f.compute_hjorth_mobility(segmento)
            complejidad = mne_f.compute_hjorth_complexity(segmento)
            matriz_caract[i][j] = np.concatenate([movilidad, complejidad])
            
    return matriz_caract.reshape([n_trials*n_segs, n_canales*num_feat])

def caracteristicas_fractales(matriz_datos):
    '''Calcula Dimensiones Fractales de Higuchi y Katz.'''
    n_trials, n_segs, n_canales, _ = matriz_datos.shape
    num_feat = 2

    matriz_caract = np.empty([n_trials, n_segs, n_canales * num_feat])
    
    for i, trial in enumerate(matriz_datos):
        for j, segmento in enumerate(trial):
            higuchi = mne_f.compute_higuchi_fd(segmento)
            katz = mne_f.compute_katz_fd(segmento)
            matriz_caract[i][j] = np.concatenate([higuchi, katz])
            
    return matriz_caract.reshape([n_trials*n_segs, n_canales*num_feat])
