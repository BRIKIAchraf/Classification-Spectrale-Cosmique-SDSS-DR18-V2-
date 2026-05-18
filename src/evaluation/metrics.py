import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

def calculate_purity(true_labels, pred_labels):
    """
    Calcule la Pureté du Clustering (Clustering Purity).
    Purity = (1 / N) * sum_k ( max_j | w_k intersect c_j | )
    
    Parameters:
    -----------
    true_labels : np.ndarray
        Labels d'origine.
    pred_labels : np.ndarray
        Labels prédits par le clustering.
        
    Returns:
    --------
    purity : float
        Score de pureté (entre 0.0 et 1.0).
    """
    # Matrice de contingence C x K via pandas crosstab
    contingency_matrix = pd.crosstab(true_labels, pred_labels).values
    # Somme des maxima de chaque colonne (chaque cluster)
    sum_max = np.sum(np.max(contingency_matrix, axis=0))
    purity = sum_max / len(true_labels)
    return purity

def evaluate_clustering(X, true_labels_str, pred_labels):
    """
    Évalue le clustering via plusieurs métriques ML standard et physiques.
    
    Parameters:
    -----------
    X : np.ndarray
        Données normalisées utilisées pour le clustering (n_objects, n_features).
    true_labels_str : np.ndarray
        Labels textuels d'origine (ex: 'SPIRAL', 'ELLIPTICAL').
    pred_labels : np.ndarray
        Clusters prédits (0, 1, 2, ...).
        
    Returns:
    --------
    metrics : dict
        Métriques de performance.
    """
    # Encoder les labels réels en entiers pour sklearn
    unique_true, true_labels = np.unique(true_labels_str, return_inverse=True)
    
    # 1. Adjusted Rand Index (ARI) - Corrige le hasard
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # 2. Silhouette Score (cohésion/séparation géométrique)
    # Requis au moins 2 clusters et n_objects > n_clusters
    n_unique_pred = len(np.unique(pred_labels))
    if 1 < n_unique_pred < len(pred_labels):
        silhouette = silhouette_score(X, pred_labels)
    else:
        silhouette = 0.0
        
    # 3. Pureté du Clustering
    purity = calculate_purity(true_labels, pred_labels)
    
    # 4. Matrice de Confusion physique C x K
    # Utilise pandas.crosstab pour obtenir la table de contingence exacte sans contrainte carrée
    conf_df = pd.crosstab(true_labels_str, pred_labels)
    
    # Assurer que toutes les colonnes de 0 à max_cluster sont présentes, même si un cluster est vide
    max_cluster = pred_labels.max() if len(pred_labels) > 0 else 0
    for i in range(max_cluster + 1):
        if i not in conf_df.columns:
            conf_df[i] = 0
            
    # Trier les colonnes pour avoir l'ordre 0, 1, 2, ...
    conf_df = conf_df.reindex(columns=range(max_cluster + 1), fill_value=0)
    conf_mat = conf_df.values
    
    return {
        'ari': ari,
        'silhouette': silhouette,
        'purity': purity,
        'confusion_matrix': conf_mat,
        'unique_true_classes': unique_true
    }

def print_scientific_report(metrics):
    """
    Affiche un rapport scientifique détaillé et propre dans la console.
    """
    print("-" * 50)
    print("           RAPPORT D'ÉVALUATION DE CLUSTERING            ")
    print("-" * 50)
    print(f"Indice de Rand Ajusté (ARI)   : {metrics['ari']:.4f}")
    print(f"Score de Silhouette           : {metrics['silhouette']:.4f}")
    print(f"Pureté du Clustering (Purity) : {metrics['purity']:.4%}")
    print("-" * 50)
    print("Matrice de Confusion (Classes Réelles vs Clusters Prédits) :")
    
    # Header des clusters
    n_clusters = metrics['confusion_matrix'].shape[1]
    cluster_headers = " ".join([f"Cluster {i:2d}" for i in range(n_clusters)])
    print(f"{'Classe SDSS':<16} | {cluster_headers}")
    print("-" * (20 + 11 * n_clusters))
    
    for i, cls in enumerate(metrics['unique_true_classes']):
        row_str = " ".join([f"{val:10d}" for val in metrics['confusion_matrix'][i]])
        print(f"{cls:<16} | {row_str}")
    print("-" * 50)
