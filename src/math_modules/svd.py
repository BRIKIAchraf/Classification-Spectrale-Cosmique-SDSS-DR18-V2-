import numpy as np

def decompose_svd(A):
    """
    Calcule la décomposition en valeurs singulières (SVD) : A = U * diag(S) * Vt.
    
    Parameters:
    -----------
    A : np.ndarray
        Matrice d'entrée de dimensions (m, n).
        
    Returns:
    --------
    U : np.ndarray
        Matrice orthogonale gauche (m, m).
    S : np.ndarray
        Vecteur des valeurs singulières triées par ordre décroissant (min(m, n),).
    Vt : np.ndarray
        Matrice orthogonale droite transposée (n, n).
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt

def randomized_svd_halko(A, k, n_iter=2, p=10):
    r"""
    Calcule la SVD randomisée (Méthode de Halko, Martinsson & Tropp 2011) à partir de zéro.
    Idéal pour les matrices spectrales massives de dimensions (m, n).
    Complexité réduite de O(mn*min(m,n)) à O(mn*log(k)).
    
    Parameters:
    -----------
    A : np.ndarray
        Matrice d'entrée de dimensions (m, n).
    k : int
        Nombre de valeurs singulières cibles.
    n_iter : int
        Nombre d'itérations de puissance (power iterations) pour affiner la projection.
    p : int
        Facteur d'oversampling pour garantir la capture des composantes clés.
        
    Returns:
    --------
    U_k : np.ndarray
        Matrice orthogonale gauche de taille (m, k).
    S_k : np.ndarray
        Vecteur des k premières valeurs singulières.
    Vt_k : np.ndarray
        Matrice orthogonale droite transposée de taille (k, n).
    """
    m, n = A.shape
    l = k + p  # Dimension du sous-espace randomisé
    
    # 1. Génération d'une matrice aléatoire gaussienne Omega
    Omega = np.random.randn(n, l)
    
    # 2. Projection de A sur le sous-espace : Y = A * Omega
    Y = np.dot(A, Omega)
    
    # 3. Orthogonalisation QR pour obtenir la base orthonormale Q
    Q, _ = np.linalg.qr(Y)
    
    # 4. Itérations de puissance pour de meilleurs résultats sur des valeurs singulières décroissantes lentement
    for _ in range(n_iter):
        # Passage par A^T pour projeter dans le domaine dual
        Q, _ = np.linalg.qr(np.dot(A.T, Q))
        # Retour au domaine d'origine
        Q, _ = np.linalg.qr(np.dot(A, Q))
        
    # Q est maintenant une excellente base orthonormale de dimensions (m, l)
    
    # 5. Projection de A sur la base réduite Q : B = Q^T * A (taille l x n)
    B = np.dot(Q.T, A)
    
    # 6. SVD standard de la matrice réduite B (petite taille)
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # 7. Reconstruction de la matrice orthogonale gauche d'origine
    U = np.dot(Q, U_tilde)
    
    # 8. Troncature aux k premières composantes
    return U[:, :k], S[:k], Vt[:k, :]

def compress_spectral_svd(U, S, Vt, k):
    """
    Réduit le bruit astrophysique en projetant la matrice sur l'espace des k premières
    composantes singulières (Approximation de rang k de Eckart-Young-Mirsky).
    
    Parameters:
    -----------
    U, S, Vt : np.ndarray
        Composantes issues de la SVD.
    k : int
        Nombre de valeurs singulières à conserver (rang de compression).
        
    Returns:
    --------
    A_compressed : np.ndarray
        Matrice débruitée reconstruite de rang k.
    """
    k = min(k, len(S))
    S_k = S[:k]
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    A_compressed = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
    return A_compressed

def compute_pseudo_inverse(U, S, Vt, rcond=1e-15):
    r"""
    Calcule le pseudo-inverse de Moore-Penrose A^\dagger = V * Sigma^\dagger * U^T.
    Utile pour la résolution optimale par moindres carrés min ||Ax - b||_2.
    
    Parameters:
    -----------
    U, S, Vt : np.ndarray
        Composantes de la SVD.
    rcond : float
        Seuil de troncature en dessous duquel les valeurs singulières sont considérées comme nulles.
        
    Returns:
    --------
    A_pseudo : np.ndarray
        Matrice pseudo-inverse de dimensions (n, m).
    """
    # Sigma^\dagger : inverse les valeurs singulières non nulles (supérieures à rcond * max(S))
    max_val = np.max(S) if len(S) > 0 else 0.0
    limit = rcond * max_val
    
    S_inv = np.zeros_like(S)
    non_zero = S > limit
    S_inv[non_zero] = 1.0 / S[non_zero]
    
    # A_dagger = V * Sigma_inv * U^T
    # Vt est déjà V^T, donc V = Vt.T
    # U est U, donc U^T = U.T
    A_pseudo = np.dot(Vt.T, np.dot(np.diag(S_inv), U.T))
    return A_pseudo

def compute_spectral_norm(A, S=None):
    """
    Calcule la norme spectrale de la matrice ||A||_2 = sigma_max = sqrt(rho(A^T A)).
    
    Parameters:
    -----------
    A : np.ndarray
        Matrice d'entrée.
    S : np.ndarray, optional
        Vecteur des valeurs singulières si déjà calculé, pour accélérer.
        
    Returns:
    --------
    norm : float
        Norme spectrale de A.
    """
    if S is None:
        _, S, _ = decompose_svd(A)
    return float(S[0]) if len(S) > 0 else 0.0

def verify_orthogonality(Q, tol=1e-12):
    """
    Vérifie numériquement la propriété d'orthogonalité Q^T * Q = I.
    Garantit la stabilité numérique et l'absence d'amplification d'erreur.
    
    Parameters:
    -----------
    Q : np.ndarray
        Matrice à tester.
    tol : float
        Tolérance de précision numérique.
        
    Returns:
    --------
    is_orthogonal : bool
        True si Q^T * Q est proche de l'identité dans la limite de tol.
    """
    n = Q.shape[1]
    QtQ = np.dot(Q.T, Q)
    I = np.eye(n)
    error = np.max(np.abs(QtQ - I))
    return error < tol, error
