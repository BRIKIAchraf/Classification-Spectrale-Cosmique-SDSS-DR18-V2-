import os
import sys

# Ajouter le chemin racine au PYTHONPATH pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import GalacticPipeline

def main():
    print("=" * 60)
    print("   SPECTRAL GRAPH THEORY · SVD (HALKO) · NUMERICAL QUADRATURE   ")
    print("      Classification Astrophysique SDSS DR18 (Architecture V2) ")
    print("=" * 60)
    
    # Initialiser le pipeline
    pipeline = GalacticPipeline(data_dir="data", output_dir="results")
    
    # Exécuter l'architecture scalable complète
    results = pipeline.run(
        n_objects=200,              # Nombre d'objets célestes à analyser (900M en production)
        k_svd=15,                   # Rang de compression spectrale (Halko SVD)
        n_quad=150,                 # Subdivision de la quadrature numérique (ordre 2)
        k_neighbors=10,             # Plus proches voisins pour le graphe k-NN creux
        rbf_gamma=1.2,              # Paramètre d'échelle du noyau Gaussien RBF
        simulated_noise_level=0.03  # Niveau de bruit instrumental Poisson + thermique simuler
    )
    
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print("                 BILAN DE L'EXÉCUTION SCIENTIFIQUE                ")
    print("=" * 60)
    print(f"[-] Nombre optimal de clusters trouvés (Eigengap) : {results['opt_k']}")
    print(f"[-] Pureté de la classification spectrale (Purity) : {metrics['purity']:.2%}")
    print(f"[-] Indice de Rand Ajusté (ARI)                   : {metrics['ari']:.4f}")
    print(f"[-] Cohérence géométrique de la Silhouette        : {metrics['silhouette']:.4f}")
    print("\nVisualisations de niveau publication scientifique exportées dans `results/` :")
    print("1. [01_svd_halko_denoising.png] - Correction Doppler (Redshift) & Débruitage par SVD Halko")
    print("2. [02_astrophysical_color_color.png] - Diagramme couleur-couleur physique (u-g vs g-r)")
    print("3. [03_sky_map_clusters.png]        - Sky Map céleste (RA/DEC) partitionnée spatialement")
    print("4. [04_tsne_spectral_clustering.png] - Projection t-SNE 2D et séparation du clustering creux")
    print("=" * 60)

if __name__ == "__main__":
    main()
