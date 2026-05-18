import os
import sys
import numpy as np
import pandas as pd
import streamlit as plt
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.sparse.linalg as spla

# Configurer la page pour une expérience premium en grand écran
st.set_page_config(
    page_title="SDSS DR18 Cosmic Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le chemin racine pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import GalacticPipeline
from src.math_modules.spectral import (
    compute_sparse_knn_graph, 
    compute_sparse_laplacian, 
    extract_sparse_fiedler_vector,
    select_n_clusters_eigengap
)
from src.math_modules.svd import randomized_svd_halko, verify_orthogonality
from src.math_modules.quadrature import extract_spectral_features
from src.astrophysics.physics_corrections import (
    apply_redshift_correction, 
    apply_galactic_extinction_ccm, 
    simulate_instrumental_noise
)
from src.evaluation.metrics import evaluate_clustering

# CSS personnalisé pour un design sombre premium et moderne (style MLOps / Observatoire)
st.markdown("""
<style>
    .reportview-container {
        background: #0f111a;
    }
    .stMetric {
        background-color: #171a26;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3440;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    h1 {
        color: #88c0d0;
        font-family: 'Outfit', sans-serif;
    }
    h3 {
        color: #d8dee9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #171a26;
        border-radius: 4px;
        color: #d8dee9;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #81a1c1 !important;
        color: #2e3440 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal avec design moderne
st.markdown("# 🔭 Observatoire Virtuel SDSS DR18")
st.markdown("### Théorie Spectrale des Graphes · SVD Halko · Quadrature Numérique")
st.write("Ajustez les paramètres astrophysiques et observez instantanément la classification et la structure de l'univers.")
st.markdown("---")

# Initialiser le pipeline
pipeline = GalacticPipeline(data_dir="data", output_dir="results")

# --- BARRE LATÉRALE DE RÉGLAGE EN TEMPS RÉEL (SIDEBAR) ---
st.sidebar.markdown("## ⚙️ Paramètres Physiques & Algorithmiques")

n_objects = st.sidebar.slider(
    "Volume d'objets célestes (N)", 
    min_value=50, 
    max_value=300, 
    value=150, 
    step=25,
    help="Nombre de galaxies/quasars à simuler du catalogue SDSS."
)

simulated_noise_level = st.sidebar.slider(
    "Niveau de bruit instrumental (S/N)", 
    min_value=0.00, 
    max_value=0.20, 
    value=0.03, 
    step=0.01,
    help="Bruit de lecture CCD et bruit de Poisson des photons."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧮 Hyperparamètres Mathématiques")

k_svd = st.sidebar.slider(
    "Rang de compression SVD (k_svd)", 
    min_value=5, 
    max_value=40, 
    value=15, 
    step=5,
    help="Nombre de composantes singulières de Halko conservées pour éliminer le bruit."
)

k_neighbors = st.sidebar.slider(
    "Voisins du graphe k-NN", 
    min_value=3, 
    max_value=25, 
    value=10, 
    step=1,
    help="Nombre de connexions par objet pour le graphe de similarité cosmique creux."
)

rbf_gamma = st.sidebar.slider(
    "Paramètre d'échelle Gamma (RBF)", 
    min_value=0.1, 
    max_value=3.0, 
    value=1.2, 
    step=0.1,
    help="Sensibilité du noyau de similarité Gaussienne."
)

# --- CACHING STRATÉGIQUE DES DONNÉES CIBLES ---
# On évite de régénérer le dataset ou de relancer Spark si le nombre d'objets n'a pas changé !
@st.cache_data
def get_cached_raw_data(n):
    # Charger ou générer les données
    data = pipeline.loader.load_cached_data()
    if data is None or len(data[0]) != n:
        metadata, wavelengths, spectra = pipeline.loader.generate_synthetic_astrophysical_data(n)
    else:
        metadata, wavelengths, spectra = data
    return metadata, wavelengths, spectra

metadata, wavelengths, spectra = get_cached_raw_data(n_objects)

# --- CALCUL RÉACTIF DU PIPELINE EN TEMPS RÉEL ---
with st.spinner("Exécution réactive de l'algèbre orthogonale (Halko SVD, Sparse Laplacian, ARPACK)..."):
    # 1. Spark Ingestion & Spatial Partitioning (simulé localement pour la vitesse du dashboard)
    metadata_clean = pipeline.big_data_pipeline.run_etl(metadata, spectra, wavelengths)
    
    # 2. Ajout du bruit
    spectra_noisy = simulate_instrumental_noise(spectra, noise_level=simulated_noise_level, noise_type='poisson')
    
    # 3. Correction Doppler (Redshift)
    redshifts = metadata_clean['z'].values
    spectra_rest, target_wavelengths = apply_redshift_correction(spectra_noisy, wavelengths, redshifts)
    
    # 4. Correction de l'extinction galactique CCM
    ebv = metadata_clean['e_b_v'].values
    extinction_factors = apply_galactic_extinction_ccm(target_wavelengths, ebv)
    spectra_physical = spectra_rest * extinction_factors
    
    # 5. SVD Randomisée de Halko (from scratch)
    U_rand, S_rand, Vt_rand = randomized_svd_halko(spectra_physical, k=k_svd, n_iter=3, p=10)
    spectra_clean = np.dot(U_rand, np.dot(np.diag(S_rand), Vt_rand))
    
    # 6. Quadrature Numérique
    features = extract_spectral_features(spectra_clean, target_wavelengths, N_quad=150)
    features_norm = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    # 7. Graphe creux k-NN & Laplacien Normalisé
    A_sparse = compute_sparse_knn_graph(features_norm, k_neighbors=k_neighbors, gamma=rbf_gamma)
    L_sparse, _ = compute_sparse_laplacian(A_sparse, normalized=True)
    
    # 8. ARPACK et Heuristique Eigengap
    eigenvals, fiedler_val, fiedler_vec = extract_sparse_fiedler_vector(L_sparse, k_eigen=8)
    opt_k, gaps = select_n_clusters_eigengap(eigenvals)
    
    # 9. Spectral Clustering via K-Means
    _, eigenvectors = np.linalg.eigh(L_sparse.toarray())
    idx = np.argsort(eigenvals)
    U_spectral = eigenvectors[:, idx[:opt_k]]
    row_norms = np.linalg.norm(U_spectral, axis=1, keepdims=True) + 1e-12
    U_spectral_normalized = U_spectral / row_norms
    
    kmeans = KMeans(n_clusters=opt_k, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(U_spectral_normalized)
    
    # 10. Évaluation
    true_classes = metadata_clean['class'].values
    metrics = evaluate_clustering(features_norm, true_classes, predicted_labels)

# --- ZONE DES INDICATEURS CLÉS (METRICS) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Clusters optimaux (Eigengap)", f"k = {opt_k}", delta=None)
with col2:
    st.metric("Pureté du Clustering (Purity)", f"{metrics['purity']:.2%}")
with col3:
    st.metric("Indice de Rand (ARI)", f"{metrics['ari']:.4f}")
with col4:
    st.metric("Cohérence Silhouette", f"{metrics['silhouette']:.4f}")

st.markdown("---")

# --- ZONE DE VISUALISATION INTERACTIVE (TABS) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 SVD & Débruitage", 
    "🎨 Diagramme Couleur-Couleur", 
    "🗺️ Sky Map RA/DEC", 
    "🌀 Projection t-SNE 2D",
    "📊 Matrice de Confusion"
])

with tab1:
    st.write("#### Débruitage spectral de rang k via la SVD Randomisée de Halko")
    fig, axes = plt.subplots(len(np.unique(true_classes)), 1, figsize=(10, 8))
    unique_classes = np.unique(true_classes)
    for i, cls in enumerate(unique_classes):
        idx = np.where(true_classes == cls)[0][0]
        ax = axes[i] if len(unique_classes) > 1 else axes
        ax.plot(target_wavelengths, spectra_noisy[idx], alpha=0.35, label=f"{cls} Bruit d'origine")
        ax.plot(target_wavelengths, spectra_clean[idx], color='#bf616a', linewidth=1.5, label=f"{cls} Corrigé + Halko SVD")
        ax.legend(loc="upper right")
        ax.set_ylabel("Flux Relatif")
        if i == 0:
            ax.set_title("Spectres Astronomiques Rest-Frame débruités")
    plt.xlabel("Longueur d'onde au repos (Å)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.write("#### Diagramme couleur-couleur SDSS (u - g vs g - r) - Vérité terrain")
    u_g = metadata_clean['u'].values - metadata_clean['g'].values
    g_r = metadata_clean['g'].values - metadata_clean['r'].values
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in unique_classes:
        mask = true_classes == cls
        ax.scatter(g_r[mask], u_g[mask], label=cls, s=35, alpha=0.7, edgecolor='k')
    ax.set_title("Distribution photométrique du ciel")
    ax.set_xlabel("Index de Couleur g - r")
    ax.set_ylabel("Index de Couleur u - g")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.write("#### Distribution céleste réelle des clusters prédits (Coordonnées RA/DEC)")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#5e81ac', '#d08770', '#a3be8c', '#bf616a', '#b48ead', '#ebcb8b', '#8fbcbb', '#4c566a']
    for cluster in np.unique(predicted_labels):
        mask = predicted_labels == cluster
        ax.scatter(
            metadata_clean['ra'].values[mask], 
            metadata_clean['dec'].values[mask], 
            label=f"Cluster {cluster}", 
            s=35, 
            color=colors[cluster % len(colors)],
            edgecolor='k',
            alpha=0.8
        )
    ax.set_title("Partitionnement spatial céleste")
    ax.set_xlabel("Ascension Droite (RA en degrés)")
    ax.set_ylabel("Déclinaison (DEC en degrés)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.write("#### Séparation géométrique dans l'espace t-SNE 2D")
    # Calcul de la projection t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    X_2d = tsne.fit_transform(features_norm)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Classes réelles
    for cls in unique_classes:
        mask = true_classes == cls
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], label=cls, s=30, alpha=0.7, edgecolor='k')
    ax1.set_title("Vérité terrain (Classes physiques SDSS)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Clusters prédits
    for cluster in np.unique(predicted_labels):
        mask = predicted_labels == cluster
        ax2.scatter(
            X_2d[mask, 0], 
            X_2d[mask, 1], 
            label=f"Cluster {cluster}", 
            s=30, 
            color=colors[cluster % len(colors)], 
            alpha=0.7, 
            edgecolor='k'
        )
    ax2.set_title("Clustering Spectral Naturel (k-NN creux)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab5:
    st.write("#### Matrice de Confusion : Classes réelles (lignes) vs Clusters prédits (colonnes)")
    
    # Création du DataFrame de confusion formaté de manière 100% robuste
    n_cols = metrics['confusion_matrix'].shape[1]
    conf_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=metrics['unique_true_classes'],
        columns=[f"Cluster {i}" for i in range(n_cols)]
    )
    
    # Affichage stylisé interactif
    st.dataframe(conf_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    
    # Analyse automatique textuelle des clusters
    st.write("##### 🔬 Analyse du partitionnement spectrale :")
    for cluster in range(opt_k):
        col_data = conf_df[f"Cluster {cluster}"]
        dominant_class = col_data.idxmax()
        percentage = col_data.max() / col_data.sum()
        st.write(f"- Le **Cluster {cluster}** est principalement composé de **{dominant_class}** avec une pureté locale de **{percentage:.1%}**.")
