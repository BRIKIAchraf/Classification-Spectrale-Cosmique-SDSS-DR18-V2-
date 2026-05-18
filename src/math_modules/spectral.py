import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors

def compute_similarity_matrix(X, gamma=1.0):
    """
    Calcule la matrice d'adjacence A en utilisant un noyau RBF (Radio Basis Function).
    A_ij = exp(-gamma * ||x_i - x_j||^2) si i != j, sinon 0.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice de données (n_objects, n_features).
    gamma : float
        Paramètre de l'échelle du noyau RBF.
        
    Returns:
    --------
    A : np.ndarray
        Matrice de similarité (d'adjacence) symétrique.
    """
    n = X.shape[0]
    # Calcul des distances euclidiennes carrées par paires
    sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    # RBF Kernel
    A = np.exp(-gamma * sq_dists)
    # Rendre les éléments diagonaux nuls (pas d'auto-boucles dans le graphe cosmique)
    np.fill_diagonal(A, 0)
    return A

def compute_laplacian(A, normalized=False):
    """
    Calcule le Laplacien spectral du graphe L = D - A.
    
    Parameters:
    -----------
    A : np.ndarray
        Matrice d'adjacence symétrique.
    normalized : bool
        Si True, calcule le Laplacien symétrique normalisé : L_sym = I - D^{-1/2} A D^{-1/2}.
        
    Returns:
    --------
    L : np.ndarray
        Matrice Laplacienne.
    D : np.ndarray
        Matrice des degrés (diagonale).
    """
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    
    if not normalized:
        L = D - A
    else:
        # Éviter la division par zéro pour les nœuds isolés
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_degrees = 1.0 / np.sqrt(degrees)
        inv_sqrt_degrees[np.isinf(inv_sqrt_degrees)] = 0.0
        D_inv_sqrt = np.diag(inv_sqrt_degrees)
        
        L = np.eye(A.shape[0]) - np.dot(D_inv_sqrt, np.dot(A, D_inv_sqrt))
        
    return L, D

def extract_fiedler_vector(L):
    """
    Calcule le vecteur de Fiedler d'un Laplacien L.
    Le vecteur de Fiedler est le vecteur propre associé à la deuxième plus petite valeur propre.
    
    Parameters:
    -----------
    L : np.ndarray
        Matrice Laplacienne symétrique.
        
    Returns:
    --------
    fiedler_val : float
        Deuxième plus petite valeur propre (connectivité algébrique du graphe).
    fiedler_vec : np.ndarray
        Vecteur de Fiedler (utilisé pour la partition spectrale).
    """
    # On résout le problème spectral Lv = lambda v
    # Puisque L est symétrique réelle positive, ses valeurs propres sont réelles >= 0.
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Tri par ordre croissant des valeurs propres
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # La première valeur propre (index 0) doit être proche de 0 (composante connexe globale)
    # Le vecteur de Fiedler est à l'index 1 (deuxième plus petite valeur propre)
    if len(eigenvalues) > 1:
        fiedler_val = eigenvalues[1]
        fiedler_vec = eigenvectors[:, 1]
    else:
        fiedler_val = eigenvalues[0]
        fiedler_vec = eigenvectors[:, 0]
        
    return fiedler_val, fiedler_vec

def spectral_partition(fiedler_vec):
    """
    Effectue une bi-partition optimale des clusters cosmiques basée sur le vecteur de Fiedler.
    
    Parameters:
    -----------
    fiedler_vec : np.ndarray
        Le vecteur de Fiedler.
        
    Returns:
    --------
    labels : np.ndarray
        Labels binaires (0 ou 1) désignant le cluster d'appartenance.
    """
    # La séparation optimale classique se fait selon le signe du vecteur de Fiedler
    # (ou par K-means à 1 dimension, mais le signe est le théorème fondamental de Fiedler).
    labels = (fiedler_vec >= 0).astype(int)
    return labels

def compute_sparse_knn_graph(X, k_neighbors=10, gamma=1.0):
    """
    Calcule une matrice d'adjacence creuse A en utilisant l'algorithme k-NN (plus proches voisins).
    L'espace mémoire requis passe de O(N²) à O(N*k), permettant de traiter des millions d'objets.
    
    Parameters:
    -----------
    X : np.ndarray
        Matrice de données (n_objects, n_features).
    k_neighbors : int
        Nombre de voisins à considérer pour chaque nœud du graphe.
    gamma : float
        Paramètre d'échelle pour le noyau Gaussien.
        
    Returns:
    --------
    A_sparse : scipy.sparse.csr_matrix
        Matrice d'adjacence creuse symétrique de taille (n_objects, n_objects).
    """
    n = X.shape[0]
    # k_neighbors ne peut pas dépasser n - 1
    k = min(k_neighbors, n - 1)
    
    # 1. Utiliser NearestNeighbors de scikit-learn pour trouver les distances
    # Cela utilise KDTree / BallTree en O(n log n) au lieu de O(n^2)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nn.fit(X)
    
    # Obtenir la matrice des distances des k plus proches voisins
    # Le premier voisin est toujours l'objet lui-même (distance 0), on le conserve ou on le filtre après.
    distances_matrix = nn.kneighbors_graph(X, mode='distance')
    
    # 2. Convertir les distances en similarités Gaussiennes : exp(-gamma * dist^2)
    # On opère directement sur les données non nulles de la matrice CSR pour l'efficacité mémoire.
    A_sparse = distances_matrix.copy()
    A_sparse.data = np.exp(-gamma * (A_sparse.data ** 2))
    
    # Rendre les éléments diagonaux nuls (pas de boucles sur soi-même)
    A_sparse.setdiag(0)
    A_sparse.eliminate_zeros()
    
    # 3. Garantir la symétrie : A = (A + A^T) / 2
    A_sparse = (A_sparse + A_sparse.T) / 2.0
    
    return A_sparse.tocsr()

def compute_sparse_laplacian(A, normalized=True):
    """
    Calcule le Laplacien spectral creux L d'un graphe à partir de sa matrice d'adjacence creuse A.
    
    Parameters:
    -----------
    A : scipy.sparse.csr_matrix
        Matrice d'adjacence creuse symétrique.
    normalized : bool
        Si True, calcule le Laplacien symétrique normalisé : L_sym = I - D^{-1/2} A D^{-1/2}.
        
    Returns:
    --------
    L_sparse : scipy.sparse.csr_matrix
        Matrice Laplacienne creuse.
    degrees : np.ndarray
        Vecteur des degrés de chaque nœud.
    """
    # Les degrés sont la somme des lignes de A
    degrees = np.array(A.sum(axis=1)).flatten()
    
    if not normalized:
        D = sp.diags(degrees)
        L_sparse = D - A
    else:
        # D^{-1/2}
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_degrees = 1.0 / np.sqrt(degrees)
        inv_sqrt_degrees[np.isinf(inv_sqrt_degrees)] = 0.0
        
        D_inv_sqrt = sp.diags(inv_sqrt_degrees)
        I = sp.eye(A.shape[0], format='csr')
        
        # L_sym = I - D_inv_sqrt * A * D_inv_sqrt
        L_sparse = I - D_inv_sqrt.dot(A).dot(D_inv_sqrt)
        
    return L_sparse.tocsr(), degrees

def extract_sparse_fiedler_vector(L, k_eigen=6):
    """
    Calcule le vecteur de Fiedler pour un Laplacien creux en résolvant le problème aux
    valeurs propres creux via le solveur ARPACK (eigsh), évitant la complexité O(N³).
    
    Parameters:
    -----------
    L : scipy.sparse.csr_matrix
        Matrice Laplacienne creuse symétrique.
    k_eigen : int
        Nombre de plus petites valeurs propres à extraire.
        
    Returns:
    --------
    eigenvalues : np.ndarray
        Les k_eigen premières valeurs propres.
    fiedler_val : float
        Deuxième plus petite valeur propre.
    fiedler_vec : np.ndarray
        Vecteur propre de Fiedler associé.
    """
    n = L.shape[0]
    # On demande les k_eigen plus petites valeurs propres (SM : Smallest Magnitude)
    # L'utilisation de sigma=0 ou d'un décalage spectral améliore la convergence d'ARPACK
    k = min(k_eigen, n - 2)
    
    # Résoudre les k plus petites valeurs propres réelles symétriques
    eigenvalues, eigenvectors = spla.eigsh(L, k=k, which='SM', tol=1e-10)
    
    # Trier par ordre croissant
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # La première doit être proche de 0 (lambda_1 ~ 0)
    # Le vecteur de Fiedler est associé à lambda_2 (index 1)
    fiedler_val = eigenvalues[1]
    fiedler_vec = eigenvectors[:, 1]
    
    return eigenvalues, fiedler_val, fiedler_vec

def select_n_clusters_eigengap(eigenvalues):
    """
    Applique l'heuristique de l'Eigengap pour déterminer automatiquement le nombre optimal
    de clusters physiques à partir des valeurs propres du Laplacien spectral.
    
    Parameters:
    -----------
    eigenvalues : np.ndarray
        Vecteur trié des plus petites valeurs propres du Laplacien creux.
        
    Returns:
    --------
    optimal_k : int
        Nombre suggéré de clusters (k).
    gaps : np.ndarray
        Les différences d'eigengap calculées (lambda_{i+1} - lambda_i).
    """
    # Calculer la différence entre les valeurs propres consécutives
    # g_i = lambda_{i+1} - lambda_i
    gaps = np.diff(eigenvalues)
    
    # On cherche l'index du plus grand gap (on ignore le premier gap de lambda_1 à lambda_2 si non significatif)
    # L'index + 1 donne le nombre de clusters (car indexé de 0)
    optimal_k = np.argmax(gaps[1:]) + 2 # Commence à chercher à partir de lambda_2
    
    return optimal_k, gaps
