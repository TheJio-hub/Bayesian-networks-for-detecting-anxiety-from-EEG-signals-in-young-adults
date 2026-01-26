import mne_features.univariate as mne_f
import numpy as np

def caracteristicas_series_tiempo(datos):
    '''
    Calcula las caracteristicas de varianza, RMS y amplitud pico-a-pico usando mne_features.

    Args:
        datos (ndarray): Datos EEG de forma (n_trials, n_secs, n_channels, n_samples).

    Returns:
        ndarray: Caracteristicas calculadas.
    '''
    n_trials, n_secs, n_canales, _ = datos.shape
    caracteristicas_por_canal = 3

    caracteristicas = np.empty([n_trials, n_secs, n_canales * caracteristicas_por_canal])
    for i, trial in enumerate(datos):
        for j, segundo in enumerate(trial):
            varianza = mne_f.compute_variance(segundo)
            rms = mne_f.compute_rms(segundo)
            ptp_Amp = mne_f.compute_ptp_amp(segundo)
            caracteristicas[i][j] = np.concatenate([varianza, rms, ptp_Amp])
    
    caracteristicas = caracteristicas.reshape(
        [n_trials*n_secs, n_canales*caracteristicas_por_canal])
    return caracteristicas


def caracteristicas_bandas_frecuencia(datos, bandas_frecuencia):
    '''
    Calcula la potencia en bandas de frecuencia (ej. delta, theta, alpha, beta, gamma).

    Args:
        datos (ndarray): Datos EEG.
        bandas_frecuencia (list o ndarray): Los límites de las bandas de frecuencia.
                     Ejemplo: [0.5, 4, 8, 13, 30, 100]

    Returns:
        ndarray: Caracteristicas calculadas.
    '''
    n_trials, n_secs, n_canales, sfreq = datos.shape
    caracteristicas_por_canal = len(bandas_frecuencia)-1

    caracteristicas = np.empty([n_trials, n_secs, n_canales * caracteristicas_por_canal])
    for i, trial in enumerate(datos):
        for j, segundo in enumerate(trial):
            psd = mne_f.compute_pow_freq_bands(
                sfreq, segundo, freq_bands=bandas_frecuencia)
            caracteristicas[i][j] = psd
            
    caracteristicas = caracteristicas.reshape(
        [n_trials*n_secs, n_canales*caracteristicas_por_canal])
    return caracteristicas


def caracteristicas_hjorth(datos):
    '''
    Calcula las caracteristicas de movilidad de Hjorth (espectro) y complejidad de Hjorth.

    Args:
        datos (ndarray): Datos EEG.

    Returns:
        ndarray: Caracteristicas calculadas.
    '''
    n_trials, n_secs, n_canales, sfreq = datos.shape
    caracteristicas_por_canal = 2

    caracteristicas = np.empty([n_trials, n_secs, n_canales * caracteristicas_por_canal])
    for i, trial in enumerate(datos):
        for j, segundo in enumerate(trial):
            mobility_spect = mne_f.compute_hjorth_mobility_spect(sfreq, segundo)
            complexity_spect = mne_f.compute_hjorth_complexity_spect(
                sfreq, segundo)
            caracteristicas[i][j] = np.concatenate([mobility_spect, complexity_spect])
            
    caracteristicas = caracteristicas.reshape(
        [n_trials*n_secs, n_canales*caracteristicas_por_canal])
    return caracteristicas


def caracteristicas_fractales(datos):
    '''
    Calcula la Dimensión Fractal de Higuchi y la Dimensión Fractal de Katz.

    Args:
        datos (ndarray): Datos EEG.

    Returns:
        ndarray: Caracteristicas calculadas.
    '''
    n_trials, n_secs, n_canales, _ = datos.shape
    caracteristicas_por_canal = 2

    caracteristicas = np.empty([n_trials, n_secs, n_canales * caracteristicas_por_canal])
    for i, trial in enumerate(datos):
        for j, segundo in enumerate(trial):
            higuchi = mne_f.compute_higuchi_fd(segundo)
            katz = mne_f.compute_katz_fd(segundo)
            caracteristicas[i][j] = np.concatenate([higuchi, katz])
            
    caracteristicas = caracteristicas.reshape(
        [n_trials*n_secs, n_canales*caracteristicas_por_canal])
    return caracteristicas


def caracteristicas_entropia(datos):
    '''
    Calcula entropía aproximada, entropía muestral, entropía espectral y entropía SVD.

    Args:
        datos (ndarray): Datos EEG.

    Returns:
        ndarray: Caracteristicas calculadas.
    '''
    n_trials, n_secs, n_canales, sfreq = datos.shape
    caracteristicas_por_canal = 4

    caracteristicas = np.empty([n_trials, n_secs, n_canales * caracteristicas_por_canal])
    for i, trial in enumerate(datos):
        for j, segundo in enumerate(trial):
            app_entropy = mne_f.compute_app_entropy(segundo)
            samp_entropy = mne_f.compute_samp_entropy(segundo)
            spect_entropy = mne_f.compute_spect_entropy(sfreq, segundo)
            svd_entropy = mne_f.compute_svd_entropy(segundo)
            caracteristicas[i][j] = np.concatenate(
                [app_entropy, samp_entropy, spect_entropy, svd_entropy])
                
    caracteristicas = caracteristicas.reshape(
        [n_trials*n_secs, n_canales*caracteristicas_por_canal])
    return caracteristicas
