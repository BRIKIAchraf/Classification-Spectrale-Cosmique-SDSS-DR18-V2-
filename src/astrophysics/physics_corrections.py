import numpy as np

def apply_redshift_correction(spectra, wavelengths, redshifts, target_wavelengths=None):
    """
    Rétablit les spectres observés dans leur référentiel au repos (Rest-Frame) :
    lambda_repos = lambda_obs / (1 + z)
    Puis ré-échantillonne tous les spectres sur une grille commune target_wavelengths
    pour permettre l'algèbre linéaire par paires.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Matrice de spectres observés (n_objects, n_wavelengths).
    wavelengths : np.ndarray
        Grille de longueurs d'onde observée.
    redshifts : np.ndarray
        Redshift z mesuré pour chaque objet.
    target_wavelengths : np.ndarray, optional
        La grille de longueur d'onde de référence commune. Si None, prend wavelengths.
        
    Returns:
    --------
    rest_spectra : np.ndarray
        Spectres corrigés du redshift et interpolés (n_objects, n_target_wavelengths).
    target_wavelengths : np.ndarray
        La grille de longueurs d'onde commune.
    """
    n_objects = spectra.shape[0]
    
    if target_wavelengths is None:
        target_wavelengths = wavelengths.copy()
        
    n_target = len(target_wavelengths)
    rest_spectra = np.zeros((n_objects, n_target))
    
    for i in range(n_objects):
        z = redshifts[i]
        spec = spectra[i]
        
        # 1. Calculer les longueurs d'onde d'origine dans le référentiel au repos
        wavelengths_repos = wavelengths / (1.0 + z)
        
        # 2. Interpoler le spectre repos sur la grille commune cible
        # En dehors de la grille cible, on extrapole avec les valeurs extrêmes
        rest_spectra[i] = np.interp(target_wavelengths, wavelengths_repos, spec, left=spec[0], right=spec[-1])
        
    return rest_spectra, target_wavelengths

def apply_galactic_extinction_ccm(wavelengths, E_B_V, R_V=3.1):
    """
    Calcule la correction d'extinction galactique (absorption par la poussière interstellaire)
    en utilisant la loi empirique de Cardelli, Clayton, and Mathis (1989) (CCM).
    A(lambda) = E(B-V) * [ a(x) * R_V + b(x) ] avec x = 1 / lambda (en micromètres).
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Longueurs d'onde en Angströms.
    E_B_V : float ou np.ndarray
        Excès de couleur E(B-V). Si np.ndarray, doit être de taille (n_objects,).
    R_V : float
        Rapport de l'extinction totale à sélective (3.1 par défaut dans la Voie Lactée).
        
    Returns:
    --------
    extinction_factors : np.ndarray
        Facteur multiplicateur de correction : 10^(0.4 * A_lambda).
        Multipliez le spectre observé par ce facteur pour retrouver le spectre intrinsèque.
    """
    # x en micromètres^-1
    x = 10000.0 / wavelengths
    
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    
    # Évaluation de la loi CCM selon le domaine spectral (optique/infrarouge/UV)
    # Domaine optique / proche IR : 1.1 <= x < 3.3
    mask_opt = (x >= 1.1) & (x < 3.3)
    y = x[mask_opt] - 1.82
    a[mask_opt] = 1.0 + 0.17699 * y - 0.50447 * (y**2) - 0.02427 * (y**3) + 0.72085 * (y**4) + 0.01979 * (y**5) - 0.7753 * (y**6) + 0.32999 * (y**7)
    b[mask_opt] = 1.41338 * y + 2.28305 * (y**2) + 1.07233 * (y**3) - 5.38434 * (y**4) - 0.62251 * (y**5) + 5.33077 * (y**6) - 2.2931 * (y**7)
    
    # Domaine infrarouge : 0.3 <= x < 1.1
    mask_ir = (x >= 0.3) & (x < 1.1)
    a[mask_ir] = 0.574 * (x[mask_ir]**1.61)
    b[mask_ir] = -0.527 * (x[mask_ir]**1.61)
    
    # A_lambda / E_B_V
    A_over_E = a * R_V + b
    
    if np.isscalar(E_B_V):
        A_lambda = E_B_V * A_over_E
        # Correction multiplicative
        correction = 10.0 ** (0.4 * A_lambda)
    else:
        # E_B_V de taille (n_objects, 1) pour diffusion par produit matriciel
        A_lambda = np.outer(E_B_V, A_over_E)
        correction = 10.0 ** (0.4 * A_lambda)
        
    return correction

def simulate_instrumental_noise(spectra, noise_level=0.05, noise_type='poisson'):
    """
    Simule de manière physique le bruit d'un télescope (CCD read-out noise + photon Poisson noise).
    
    Parameters:
    -----------
    spectra : np.ndarray
        Matrice de spectres propres.
    noise_level : float
        Échelle générale du bruit.
    noise_type : str
        'gaussian' (bruit thermique de lecture) ou 'poisson' (bruit de comptage de photons).
        
    Returns:
    --------
    noisy_spectra : np.ndarray
        Spectres bruités.
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0.0, noise_level, spectra.shape)
        return np.clip(spectra + noise, 1e-5, None)
    
    elif noise_type == 'poisson':
        # Le bruit de Poisson dépend du signal. Plus le flux est élevé, plus l'écart-type est grand,
        # mais plus le rapport signal sur bruit (S/N) s'améliore (S/N ~ sqrt(signal)).
        # Simulé en convertissant le flux relatif en "électrons" fictifs accumulés par le CCD.
        scale_factor = 1000.0 / noise_level
        electrons = np.clip(spectra * scale_factor, 0, None)
        noisy_electrons = np.random.poisson(electrons)
        
        # Reconversion en flux
        noisy_spectra = noisy_electrons / scale_factor
        return np.clip(noisy_spectra, 1e-5, None)
        
    return spectra
