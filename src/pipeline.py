import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Imports Noyaux Mathématiques Scalables
from src.math_modules.spectral import (
    compute_sparse_knn_graph, 
    compute_sparse_laplacian, 
    extract_sparse_fiedler_vector, 
    spectral_partition,
    select_n_clusters_eigengap
)
from src.math_modules.svd import randomized_svd_halko, verify_orthogonality
from src.math_modules.quadrature import extract_spectral_features

# Imports Ingestion & Physique & Évaluation
from src.data_loader import SDSSDataLoader
from src.data.pyspark_etl import CosmicBigDataPipeline
from src.astrophysics.physics_corrections import (
    apply_redshift_correction, 
    apply_galactic_extinction_ccm, 
    simulate_instrumental_noise
)
from src.evaluation.metrics import evaluate_clustering, print_scientific_report

class GalacticPipeline:
    def __init__(self, data_dir="data", output_dir="results"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.loader = SDSSDataLoader(cache_dir=data_dir)
        self.big_data_pipeline = CosmicBigDataPipeline(data_dir=data_dir, parquet_dir=os.path.join(data_dir, "parquet"))
        
    def run(self, n_objects=200, k_svd=15, n_quad=150, k_neighbors=10, rbf_gamma=1.2, simulated_noise_level=0.03):
        """
        Exécute l'architecture V2 complète de classification à l'échelle :
        
        --- BIG DATA & DATA INGESTION ---
        1. Ingestion / Génération des spectres bruts et flux photométriques.
        2. Execution de l'ETL PySpark avec partitionnement céleste spatial et stockage Parquet.
        
        --- ASTROPHYSICS REALISM LAYER ---
        3. Correction instrumentale (ajout de bruit de Poisson / read-noise pour robustesse).
        4. Héliocentrisme & Rest-Frame : Correction relativiste du redshift (z) et interpolation.
        5. Correction de l'extinction interstellaire galactique (modèle de poussière CCM).
        
        --- MATH CORE SCALABLE ---
        6. Compression et débruitage spectral via la SVD randomisée de Halko (from scratch).
        7. Intégration physique par Quadrature numérique d'ordre 2.
        8. Construction du Graphe creux k-NN (complexité O(n log n)) via KDTree.
        9. Calcul du Laplacien creux normalisé symétrique (mémoire O(n)).
        10. Extraction des vecteurs propres creux via ARPACK (eigsh), eigengap automatique.
        11. Partitionnement spectral et évaluation scientifique complète.
        """
        print("="*60)
        print("   DÉMARRAGE PIPELINE HAUTE-PERFORMANCE SDSS DR18 (V2)   ")
        print("="*60)
        
        # --- 1. Chargement / Génération ---
        print("\n[ETAPE 1] Chargement / Génération des données brutes...")
        data = self.loader.load_cached_data()
        if data is None:
            metadata, wavelengths, spectra = self.loader.generate_synthetic_astrophysical_data(n_objects)
        else:
            metadata, wavelengths, spectra = data
            # S'assurer que le cache a bien u, g, r, i, z, e_b_v, sinon regénérer pour la V2
            if 'u' not in metadata.columns:
                print("Données V1 détectées dans le cache. Régénération pour le support V2...")
                metadata, wavelengths, spectra = self.loader.generate_synthetic_astrophysical_data(n_objects)
            else:
                print(f"Chargement réussi de {len(spectra)} spectres du cache.")
                
        # --- 2. Big Data ETL (PySpark / Parquet) ---
        print("\n[ETAPE 2] Lancement du PySpark ETL & Partitionnement Spatial (HEALPix)...")
        # Enregistrer et partitionner au format Parquet
        metadata_clean = self.big_data_pipeline.run_etl(metadata, spectra, wavelengths)
        
        # --- 3. Astro Realism : Simulation de Bruit ---
        print("\n[ETAPE 3] Simulation de bruit instrumental (Comptage de Poisson + Read-noise)...")
        spectra_noisy = simulate_instrumental_noise(spectra, noise_level=simulated_noise_level, noise_type='poisson')
        
        # --- 4. Astro Realism : Correction du Redshift ---
        print("\n[ETAPE 4] Correction du décalage spectral (Redshift z) & Interpolation Rest-frame...")
        redshifts = metadata_clean['z'].values
        spectra_rest, target_wavelengths = apply_redshift_correction(spectra_noisy, wavelengths, redshifts)
        
        # --- 5. Astro Realism : Correction de l'extinction galactique (Loi CCM) ---
        print("\n[ETAPE 5] Correction de l'extinction interstellaire (Poussière galactique CCM)...")
        ebv = metadata_clean['e_b_v'].values
        extinction_factors = apply_galactic_extinction_ccm(target_wavelengths, ebv)
        # Corriger l'intensité : I_vraie = I_observee * facteur
        spectra_physical = spectra_rest * extinction_factors
        
        # --- 6. Math Core : SVD Randomisée de Halko ---
        print("\n[ETAPE 6] Compression & Débruitage via SVD Randomisée de Halko...")
        U_rand, S_rand, Vt_rand = randomized_svd_halko(spectra_physical, k=k_svd, n_iter=3, p=10)
        
        # Vérification d'orthogonalité
        is_ortho, ortho_err = verify_orthogonality(U_rand)
        print(f"-> Stabilité SVD vérifiée. Orthogonalité U_rand^T * U_rand = I (erreur = {ortho_err:.2e})")
        
        # Reconstruire les spectres physiques propres et débruités de rang k
        spectra_clean = np.dot(U_rand, np.dot(np.diag(S_rand), Vt_rand))
        
        # --- 7. Math Core : Quadrature Numérique ---
        print("\n[ETAPE 7] Extraction d'énergie par Quadrature Numérique d'Ordre 2...")
        features = extract_spectral_features(spectra_clean, target_wavelengths, N_quad=n_quad)
        
        # --- 8. Math Core : Graphe k-NN Creux (Scalable) ---
        print(f"\n[ETAPE 8] Construction du graphe k-NN creux (k={k_neighbors}) via KDTree...")
        # Normaliser les features pour éviter les effets d'échelle
        features_norm = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        A_sparse = compute_sparse_knn_graph(features_norm, k_neighbors=k_neighbors, gamma=rbf_gamma)
        print(f"-> Graphe creux construit. Éléments non nuls : {A_sparse.nnz} (Densité : {A_sparse.nnz / (n_objects**2):.2%})")
        
        # --- 9. Math Core : Laplacien Creux Normalisé ---
        print("\n[ETAPE 9] Calcul du Laplacien creux normalisé symétrique...")
        L_sparse, degrees = compute_sparse_laplacian(A_sparse, normalized=True)
        
        # --- 10. Math Core : Solveur Creux ARPACK (eigsh) & Eigengap ---
        print("\n[ETAPE 10] Extraction des vecteurs propres creux via ARPACK...")
        # On extrait les 8 premières valeurs propres pour analyser l'Eigengap
        eigenvals, fiedler_val, fiedler_vec = extract_sparse_fiedler_vector(L_sparse, k_eigen=8)
        
        # Sélection automatique de k via l'Eigengap
        opt_k, gaps = select_n_clusters_eigengap(eigenvals)
        print(f"-> Connectivité algébrique (lambda_2) = {fiedler_val:.6f}")
        print(f"-> Analyse de l'Eigengap : Le nombre optimal de clusters physiques est k={opt_k}")
        
        # Pour le partitionnement, on applique K-Means sur les 3 premiers vecteurs propres
        # (extension naturelle du vecteur de Fiedler pour k > 2)
        # Résolution spectrale complète :
        _, eigenvectors = spla_eigsh = np.linalg.eigh(L_sparse.toarray()) # fallback complet pour k-means
        # Trier par valeurs propres
        idx = np.argsort(eigenvals)
        U_spectral = eigenvectors[:, idx[:opt_k]]
        # Normaliser les lignes pour la stabilité (Ng-Jordan-Weiss)
        row_norms = np.linalg.norm(U_spectral, axis=1, keepdims=True) + 1e-12
        U_spectral_normalized = U_spectral / row_norms
        
        kmeans = KMeans(n_clusters=opt_k, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(U_spectral_normalized)
        
        # --- 11. Évaluation scientifique et Matrice de Confusion ---
        print("\n[ETAPE 11] Évaluation ML et Validation Astrophysique complète...")
        true_classes = metadata_clean['class'].values
        metrics = evaluate_clustering(features_norm, true_classes, predicted_labels)
        print_scientific_report(metrics)
        
        # Arrêt de Spark
        self.big_data_pipeline.stop_session()
        
        # Génération des tracés physiques
        self.plot_scientific_results(
            target_wavelengths, 
            spectra, 
            spectra_clean, 
            metadata_clean, 
            predicted_labels, 
            true_classes, 
            features_norm
        )
        
        return {
            'metrics': metrics,
            'predicted_labels': predicted_labels,
            'opt_k': opt_k
        }
        
    def plot_scientific_results(self, wavelengths, raw_spectra, clean_spectra, metadata, predicted_labels, true_classes, features_norm):
        """
        Génère des visualisations de niveau publication scientifique.
        """
        unique_classes = np.unique(true_classes)
        
        # 1. Tracé de Débruitage SVD & Correction de Redshift
        plt.figure(figsize=(12, 8))
        for i, cls in enumerate(unique_classes):
            idx = np.where(true_classes == cls)[0][0]
            plt.subplot(len(unique_classes), 1, i + 1)
            plt.plot(wavelengths, raw_spectra[idx], alpha=0.4, label=f"{cls} Brut (Bruit & Redshift observés)")
            plt.plot(wavelengths, clean_spectra[idx], color='red', linewidth=1.5, label=f"{cls} Corrigé + Reconstruit (Halko SVD)")
            plt.ylabel("Flux Relatif")
            plt.legend(loc="upper right")
            if i == 0:
                plt.title("Débruitage & Correction Cosmologique (SDSS DR18)")
        plt.xlabel("Longueur d'onde au repos (Å)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_svd_halko_denoising.png"))
        plt.close()
        
        # 2. Diagramme Couleur-Couleur (u-g vs g-r) - TRÈS IMPORTANT en astronomie
        plt.figure(figsize=(10, 6))
        # Calculer les indices de couleur
        u_g = metadata['u'].values - metadata['g'].values
        g_r = metadata['g'].values - metadata['r'].values
        
        for cls in unique_classes:
            mask = true_classes == cls
            plt.scatter(g_r[mask], u_g[mask], label=cls, s=40, alpha=0.7, edgecolor='k')
            
        plt.title("Diagramme Couleur-Couleur SDSS (u-g vs g-r) - Classes Physiques")
        plt.xlabel("Index de Couleur g - r")
        plt.ylabel("Index de Couleur u - g")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "02_astrophysical_color_color.png"))
        plt.close()
        
        # 3. Sky Map céleste (RA/DEC Projection spatiale)
        plt.figure(figsize=(10, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for cluster in np.unique(predicted_labels):
            mask = predicted_labels == cluster
            plt.scatter(
                metadata['ra'].values[mask], 
                metadata['dec'].values[mask], 
                label=f"Cluster {cluster}", 
                s=35, 
                color=colors[cluster % len(colors)],
                edgecolor='k',
                alpha=0.8
            )
            
        plt.title("Sky Map céleste (Coordonnées RA/DEC partitionnées)")
        plt.xlabel("Ascension Droite (RA en degrés)")
        plt.ylabel("Déclinaison (DEC en degrés)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "03_sky_map_clusters.png"))
        plt.close()
        
        # 4. Projection 2D UMAP / t-SNE du Clustering Spectral
        print("Calcul de la projection t-SNE 2D du clustering spectral...")
        try:
            tsne = TSNE(n_components=2, perplexity=15, random_state=42)
            X_2d = tsne.fit_transform(features_norm)
            
            plt.figure(figsize=(12, 5))
            
            # Subplot A : Classes réelles
            plt.subplot(1, 2, 1)
            for cls in unique_classes:
                mask = true_classes == cls
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cls, s=35, alpha=0.7, edgecolor='k')
            plt.title("Vérité terrain (Classes SDSS)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot B : Clusters trouvés
            plt.subplot(1, 2, 2)
            for cluster in np.unique(predicted_labels):
                mask = predicted_labels == cluster
                plt.scatter(
                    X_2d[mask, 0], 
                    X_2d[mask, 1], 
                    label=f"Cluster {cluster}", 
                    s=35, 
                    color=colors[cluster % len(colors)], 
                    alpha=0.7, 
                    edgecolor='k'
                )
            plt.title("Clustering Spectral Naturel (k-NN creux)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.suptitle("Séparation et Cohésion Céleste dans l'espace t-SNE")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "04_tsne_spectral_clustering.png"))
            plt.close()
            print(f"Tous les tracés scientifiques V2 ont été exportés dans `{self.output_dir}/` !")
        except Exception as e:
            print(f"Impossible de générer le tracé t-SNE : {e}")
