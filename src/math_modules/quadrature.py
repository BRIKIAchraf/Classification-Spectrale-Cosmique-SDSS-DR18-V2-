import numpy as np

def compute_midpoint_quadrature(f_vals, lmbda_min, lmbda_max):
    r"""
    Calcule l'approximation numérique de l'énergie spectrale intégrée
    I = \int_{lambda_min}^{lambda_max} f(lambda) dlambda
    en utilisant la méthode des points milieux (Ordre 2, erreur O(dlambda^2)).
    
    Parameters:
    -----------
    f_vals : np.ndarray
        Valeurs du flux ou de l'intensité du spectre discrétisé f(lambda_i).
        De taille N (le nombre de bins).
    lmbda_min : float
        Longueur d'onde minimale.
    lmbda_max : float
        Longueur d'onde maximale.
        
    Returns:
    --------
    integrated_energy : float
        Énergie lumineuse totale approximée.
    d_lmbda : float
        Le pas d'intégration dlambda.
    """
    N = len(f_vals)
    if N == 0:
        return 0.0, 0.0
        
    # Calcul de dlambda
    d_lmbda = (lmbda_max - lmbda_min) / N
    
    # Méthode des points milieux :
    # Si le spectre est déjà échantillonné sur les points milieux :
    # lambda_i = lambda_min + (i + 0.5) * dlambda
    # L'intégrale est simplement la somme des f(lambda_i) * dlambda
    integrated_energy = np.sum(f_vals) * d_lmbda
    
    return float(integrated_energy), float(d_lmbda)

def interpolate_spectrum_to_midpoints(lmbda_original, f_original, N_target):
    """
    Interpole un spectre physique échantillonné arbitrairement vers une grille de
    points milieux équidistants pour appliquer la quadrature d'ordre 2.
    
    Parameters:
    -----------
    lmbda_original : np.ndarray
        Longueurs d'onde du spectre d'origine.
    f_original : np.ndarray
        Flux d'origine.
    N_target : int
        Nombre de subdivisions cibles pour la quadrature.
        
    Returns:
    --------
    lmbda_midpoints : np.ndarray
        Grille de longueurs d'onde des points milieux.
    f_midpoints : np.ndarray
        Flux interpolé aux points milieux.
    """
    lmbda_min = np.min(lmbda_original)
    lmbda_max = np.max(lmbda_original)
    
    d_lmbda = (lmbda_max - lmbda_min) / N_target
    
    # Construction des points milieux
    lmbda_midpoints = lmbda_min + (np.arange(N_target) + 0.5) * d_lmbda
    
    # Interpolation linéaire du flux aux points milieux
    f_midpoints = np.interp(lmbda_midpoints, lmbda_original, f_original)
    
    return lmbda_midpoints, f_midpoints

def extract_spectral_features(spectra, wavelengths, N_quad=100):
    """
    Calcule des caractéristiques physiques d'une série de spectres en utilisant la quadrature.
    Par exemple : Énergie totale, énergie par sous-bandes (U, G, R, I, Z du SDSS).
    
    Parameters:
    -----------
    spectra : np.ndarray
        Matrice de spectres de taille (n_objects, n_wavelengths).
    wavelengths : np.ndarray
        Grille de longueurs d'onde d'origine.
    N_quad : int
        Nombre de points d'intégration pour la quadrature.
        
    Returns:
    --------
    features : np.ndarray
        Matrice de caractéristiques (n_objects, n_features).
    """
    n_objects = spectra.shape[0]
    l_min, l_max = np.min(wavelengths), np.max(wavelengths)
    
    # On définit des sous-bandes spectrales simplifiées type SDSS (en Angströms)
    # SDSS bands: u (~3543), g (~4770), r (~6231), i (~7625), z (~9134)
    bands = {
        'total': (l_min, l_max),
        'band_blue': (l_min, l_min + (l_max - l_min) * 0.33),
        'band_medium': (l_min + (l_max - l_min) * 0.33, l_min + (l_max - l_min) * 0.66),
        'band_red': (l_min + (l_max - l_min) * 0.66, l_max)
    }
    
    features = []
    
    for i in range(n_objects):
        obj_features = []
        spec = spectra[i]
        
        for band_name, (b_min, b_max) in bands.items():
            # Interpolation sur la grille des points milieux de la sous-bande
            l_mid, f_mid = interpolate_spectrum_to_midpoints(wavelengths, spec, N_quad)
            # Sélectionner uniquement les points dans la bande
            mask = (l_mid >= b_min) & (l_mid <= b_max)
            f_band = f_mid[mask]
            
            # Calcul de la quadrature sur la bande
            energy, _ = compute_midpoint_quadrature(f_band, b_min, b_max)
            obj_features.append(energy)
            
        features.append(obj_features)
        
    return np.array(features)
