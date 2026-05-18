import os
import sys
import numpy as np

# Ajouter la racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.math_modules.spectral import (
    compute_sparse_knn_graph, 
    compute_sparse_laplacian, 
    extract_sparse_fiedler_vector
)
from src.math_modules.svd import randomized_svd_halko, verify_orthogonality
from src.math_modules.quadrature import compute_midpoint_quadrature
from src.astrophysics.physics_corrections import apply_redshift_correction, apply_galactic_extinction_ccm

def test_randomized_svd_halko_properties():
    """
    Vérifie les propriétés mathématiques fondamentales de l'algorithme de Halko (SVD randomisée) :
    - L'orthogonalité des vecteurs singuliers de projection (U^T * U = I).
    - La précision de l'approximation de rang k par rapport à une SVD standard.
    """
    np.random.seed(42)
    # Créer une matrice de rang faible (physiquement similaire à un jeu de spectres compressibles)
    U_true, _ = np.linalg.qr(np.random.randn(50, 10))
    S_true = np.diag([100.0, 80.0, 50.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    Vt_true, _ = np.linalg.qr(np.random.randn(30, 10))
    # Matrice m=50, n=30 de rang numérique k=10
    A = np.dot(U_true, np.dot(S_true, Vt_true.T))
    
    k = 5
    U_rand, S_rand, Vt_rand = randomized_svd_halko(A, k=k, n_iter=3, p=5)
    
    # 1. Vérifier les dimensions
    assert U_rand.shape == (50, k)
    assert S_rand.shape == (k,)
    assert Vt_rand.shape == (k, 30)
    
    # 2. Vérifier l'orthogonalité U^T * U = I
    is_ortho, err = verify_orthogonality(U_rand)
    assert is_ortho, f"U n'est pas orthogonale dans la SVD randomisée, erreur = {err}"
    
    # 3. Vérifier que la norme des valeurs singulières capturées est proche des vraies premières valeurs singulières
    np.testing.assert_allclose(S_rand, [100.0, 80.0, 50.0, 10.0, 5.0], rtol=1e-1)

def test_sparse_graph_laplacian():
    """
    Vérifie la validité des graphes creux et du solveur ARPACK :
    - La matrice d'adjacence k-NN creuse doit être symétrique.
    - La somme des lignes du Laplacien creux non normalisé doit être nulle.
    - La première valeur propre du Laplacien doit être nulle, la seconde (Fiedler) positive.
    """
    np.random.seed(42)
    # 30 objets dans un espace 2D constituant un seul cluster connecté
    X = np.random.normal(0, 0.2, (30, 2))
    
    # 1. k-NN creux
    A_sparse = compute_sparse_knn_graph(X, k_neighbors=5, gamma=1.0)
    
    # Symétrie vérifiée
    diff = A_sparse - A_sparse.T
    assert diff.nnz == 0, "La matrice d'adjacence k-NN creuse n'est pas symétrique."
    
    # 2. Laplacien creux non normalisé
    L_sparse, degrees = compute_sparse_laplacian(A_sparse, normalized=False)
    row_sums = np.array(L_sparse.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)
    
    # 3. Solveur creux ARPACK (eigsh)
    # Laplacien normalisé pour le partitionnement
    L_norm, _ = compute_sparse_laplacian(A_sparse, normalized=True)
    eigenvals, fiedler_val, fiedler_vec = extract_sparse_fiedler_vector(L_norm, k_eigen=4)
    
    # lambda_1 ~ 0
    assert np.abs(eigenvals[0]) < 1e-10
    # lambda_2 (Fiedler) doit être > 0 (car le graphe est connexe)
    assert fiedler_val > 0.0
    assert len(fiedler_vec) == 30

def test_astrophysics_corrections():
    """
    Vérifie la justesse des calculs physiques :
    - Correction relativiste du redshift (l'onde au repos doit être décalée à gauche si z > 0).
    - Extinction galactique : les facteurs de correction doivent être supérieurs à 1.0 (le flux
      observé est toujours affaibli par la poussière interstellaire).
    """
    wavelengths = np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0])
    spectra = np.array([[1.0, 1.2, 1.5, 1.3, 1.0]])
    redshifts = np.array([0.25]) # 25% de redshift (Doppler)
    
    # 1. Redshift Correction
    spectra_rest, target_waves = apply_redshift_correction(spectra, wavelengths, redshifts)
    # A z=0.25, la raie observée à 5000 Å au repos correspondait à 5000 / 1.25 = 4000 Å
    # Le flux interpolé au repos à 4000 Å sur la grille commune doit correspondre au flux observé à 5000 Å (1.2)
    np.testing.assert_allclose(spectra_rest[0, 0], 1.2, rtol=1e-1)
    
    # 2. Extinction galactique CCM
    # Excès de couleur E(B-V) = 0.1
    correction_factors = apply_galactic_extinction_ccm(wavelengths, E_B_V=0.1)
    # La poussière interstellaire absorbe la lumière, donc la correction multiplicative doit être > 1.0 pour restaurer le flux intrinsèque
    assert np.all(correction_factors > 1.0)
    # L'absorption est plus forte dans le bleu (longueurs d'onde courtes, ex: 4000 Å) que le rouge (8000 Å)
    assert correction_factors[0] > correction_factors[-1]

def test_quadrature_precision():
    """
    Valide l'approximation numérique d'ordre 2 de la quadrature sur une fonction connue.
    """
    lmbda_min = 0.0
    lmbda_max = np.pi
    N = 150
    d_lmbda = (lmbda_max - lmbda_min) / N
    midpoints = lmbda_min + (np.arange(N) + 0.5) * d_lmbda
    
    f_vals = np.sin(midpoints)
    integrated_energy, _ = compute_midpoint_quadrature(f_vals, lmbda_min, lmbda_max)
    
    np.testing.assert_allclose(integrated_energy, 2.0, rtol=1e-4)
