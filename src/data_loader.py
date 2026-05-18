import os
import numpy as np
import pandas as pd
try:
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False

class SDSSDataLoader:
    def __init__(self, cache_dir="data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def fetch_real_sdss_data(self, limit=100):
        """
        Télécharge de réelles données spectrales et photométriques de SDSS DR18 via astroquery.
        """
        if not ASTROQUERY_AVAILABLE:
            raise ImportError("astroquery et astropy sont requis pour télécharger des données réelles de SDSS.")
            
        print(f"Connexion au serveur SDSS DR18 pour récupérer {limit} objets...")
        
        # Requête SQL pour obtenir des objets avec classification et redshifts valides
        sql_query = f"""
        SELECT TOP {limit}
            s.specobjid, s.ra, s.dec, s.class, s.subclass, s.z,
            p.u, p.g, p.r, p.i, p.z as z_mag
        FROM SpecObj AS s
        JOIN PhotoObj AS p ON s.bestobjid = p.objid
        WHERE s.class IN ('GALAXY', 'QSO', 'STAR')
          AND s.zWarning = 0
          AND s.z > 0
        """
        try:
            # Lancement de la requête SQL SDSS
            result = SDSS.query_sql(sql_query, data_release=18)
            if result is None or len(result) == 0:
                raise ValueError("La requête SDSS DR18 n'a retourné aucun résultat.")
                
            df = result.to_pandas()
            print(f"Téléchargement réussi de {len(df)} lignes de métadonnées.")
            
            # Sauvegarder dans le cache local
            df.to_csv(os.path.join(self.cache_dir, "sdss_metadata.csv"), index=False)
            return df
            
        except Exception as e:
            print(f"Erreur de connexion à SDSS DR18 : {e}")
            raise e

    def generate_synthetic_astrophysical_data(self, n_objects=150, n_wavelengths=500):
        """
        Générateur résilient de données physiques synthétiques imitant le SDSS DR18.
        Génère des spectres réalistes avec continuum de corps noir, raies d'émission/absorption
        décalées par l'effet Doppler relativiste (redshift z) selon la classe d'objet :
        - Galaxies Spirales (raies d'émission jeunes, H-alpha fortes, z modéré)
        - Galaxies Elliptiques (vieilles étoiles, absorption forte, continuum rouge, z modéré)
        - Galaxies Irrégulières (turbulentes, raies d'émission fortes, continuum bleu)
        - Quasars / QSO (très haut redshift z, raies de Lyman-alpha extrêmement élargies)
        - Objets Compacts / Étoiles (continuum thermique, absorption fine, redshift nul z~0)
        """
        print(f"Génération de {n_objects} spectres astronomiques synthétiques hautement réalistes...")
        
        # Définition de la grille de longueurs d'onde (3800 à 9200 Angströms, domaine optique SDSS)
        wavelengths = np.linspace(3800, 9200, n_wavelengths)
        
        # Classes d'objets astronomiques
        classes = ['SPIRAL', 'ELLIPTICAL', 'IRREGULAR', 'QSO', 'COMPACT_STAR']
        weights = [0.3, 0.3, 0.15, 0.15, 0.1] # Distribution typique d'observation
        
        object_classes = np.random.choice(classes, size=n_objects, p=weights)
        
        spectra = np.zeros((n_objects, n_wavelengths))
        redshifts = np.zeros(n_objects)
        
        # Caractéristiques spectrales de base des raies (longueurs d'onde au repos en Å)
        rest_lines = {
            'H_alpha': 6563.0,   # Émission hydrogène (rouge)
            'H_beta': 4861.0,    # Émission/absorption hydrogène
            'OIII_1': 5007.0,    # Oxygène doublement ionisé (galaxies actives)
            'OIII_2': 4959.0,
            'NII': 6584.0,       # Azote ionisé
            'Lyman_alpha': 1216.0, # Pour les Quasars (visible dans l'optique uniquement si z élevé !)
            'CIV': 1549.0,       # Carbone ionisé pour quasars
            'MgII': 2798.0,      # Magnésium pour quasars/étoiles
            'Na_D': 5892.0,      # Sodium (absorption froide, étoiles/elliptiques)
            'Ca_HK_1': 3933.0,   # Calcium (absorption forte)
            'Ca_HK_2': 3968.0
        }
        
        for i, obj_class in enumerate(object_classes):
            # 1. Définir le Redshift (décalage spectral z)
            if obj_class == 'COMPACT_STAR':
                z = np.random.normal(0.0, 0.0001) # Les étoiles de la Voie Lactée ne bougent pas beaucoup
            elif obj_class == 'QSO':
                z = np.random.uniform(1.5, 4.0)   # Très lointains quasars (Lyman-alpha décalé dans l'optique)
            else:
                z = np.random.uniform(0.01, 0.4)  # Galaxies locales à intermédiaires
                
            redshifts[i] = z
            
            # 2. Continuum thermique (Loi de Planck simplifiée ou pente de puissance)
            # Température de couleur effective
            if obj_class == 'ELLIPTICAL':
                temp = np.random.uniform(3000, 5000) # Étoiles géantes rouges dominantes (froid)
            elif obj_class in ['SPIRAL', 'IRREGULAR']:
                temp = np.random.uniform(6000, 10000) # Mélange jeune/vieux
            elif obj_class == 'QSO':
                temp = np.random.uniform(15000, 30000) # Disque d'accrétion ultra chaud
            else:
                temp = np.random.uniform(4000, 12000) # Étoiles diverses
                
            # Formule de Wien/Planck simplifiée pour le continuum : C(lambda)
            # f_cont = (1/lambda^5) / (exp(hc/lambda k T) - 1)
            # Approximée par une fonction puissance/thermique
            h_c_k = 1.43877e8 # constante physique pour lambda en Angströms
            continuum = (3800.0 / wavelengths)**4 * (1.0 / (np.exp(h_c_k / (wavelengths * temp)) - 0.9))
            # Normalisation
            continuum = continuum / np.max(continuum) * np.random.uniform(0.5, 2.0)
            
            # 3. Ajouter les raies spectrales décalées par le Redshift : lambda_obs = lambda_repos * (1 + z)
            spec = continuum.copy()
            
            # Fonction pour ajouter une raie gaussienne
            def add_line(w_repos, amplitude, width, is_emission=True):
                w_obs = w_repos * (1.0 + z)
                # Si la raie tombe dans la fenêtre optique [3800, 9200]
                if 3800 <= w_obs <= 9200:
                    profile = np.exp(-0.5 * ((wavelengths - w_obs) / width)**2)
                    factor = 1.0 if is_emission else -1.0
                    return amplitude * profile * factor
                return 0.0
                
            if obj_class == 'SPIRAL':
                # Galaxies spirales : Étoiles jeunes en formation -> Émission H-alpha, H-beta, OIII
                spec += add_line(rest_lines['H_alpha'], 1.2, 12.0, is_emission=True)
                spec += add_line(rest_lines['H_beta'], 0.4, 10.0, is_emission=True)
                spec += add_line(rest_lines['OIII_1'], 0.5, 10.0, is_emission=True)
                spec += add_line(rest_lines['OIII_2'], 0.17, 10.0, is_emission=True)
                spec += add_line(rest_lines['Ca_HK_1'], 0.2, 15.0, is_emission=False) # Petites absorptions
                spec += add_line(rest_lines['Ca_HK_2'], 0.2, 15.0, is_emission=False)
                
            elif obj_class == 'ELLIPTICAL':
                # Vieilles étoiles -> Fortes absorptions (Ca H&K, Na D, Mg)
                spec += add_line(rest_lines['Ca_HK_1'], 0.6, 18.0, is_emission=False)
                spec += add_line(rest_lines['Ca_HK_2'], 0.5, 18.0, is_emission=False)
                spec += add_line(rest_lines['Na_D'], 0.4, 15.0, is_emission=False)
                spec += add_line(rest_lines['MgII'], 0.3, 14.0, is_emission=False)
                
            elif obj_class == 'IRREGULAR':
                # Galaxies irrégulières / Starburst -> Émission H-alpha et OIII massives
                spec += add_line(rest_lines['H_alpha'], 2.5, 15.0, is_emission=True)
                spec += add_line(rest_lines['H_beta'], 0.8, 12.0, is_emission=True)
                spec += add_line(rest_lines['OIII_1'], 2.0, 12.0, is_emission=True)
                spec += add_line(rest_lines['OIII_2'], 0.67, 12.0, is_emission=True)
                
            elif obj_class == 'QSO':
                # Quasar : raies d'émission larges (Lyman alpha, CIV, MgII) très fortement décalées
                # Lyman alpha au repos à 1216 Å décalé à z=3.0 -> 1216 * 4 = 4864 Å !
                spec += add_line(rest_lines['Lyman_alpha'], 3.0, 60.0, is_emission=True) # Raie ultra large (effet Doppler de gaz rapides)
                spec += add_line(rest_lines['CIV'], 2.0, 50.0, is_emission=True)
                spec += add_line(rest_lines['MgII'], 1.2, 40.0, is_emission=True)
                
            elif obj_class == 'COMPACT_STAR':
                # Étoile compacte : Raies d'absorption très fines de l'hydrogène (série de Balmer)
                spec += add_line(rest_lines['H_alpha'], 0.7, 5.0, is_emission=False)
                spec += add_line(rest_lines['H_beta'], 0.8, 5.0, is_emission=False)
                spec += add_line(rest_lines['Na_D'], 0.5, 4.0, is_emission=False)
                
            # 4. Ajout de bruit astrophysique aléatoire (thermique + bruit de lecture du CCD du télescope)
            noise = np.random.normal(0.0, 0.08, n_wavelengths)
            spec = np.clip(spec + noise, 0.01, None) # Pas de flux négatif physique
            
            spectra[i] = spec
            
        # 5. Création des métadonnées photométriques physiques cohérentes (u, g, r, i, z) et de l'extinction E_B_V
        u_mags = np.zeros(n_objects)
        g_mags = np.zeros(n_objects)
        r_mags = np.zeros(n_objects)
        i_mags = np.zeros(n_objects)
        z_mags = np.zeros(n_objects)
        e_b_v = np.random.exponential(scale=0.05, size=n_objects) + 0.01 # Extinction galactique réaliste
        
        for idx, obj_class in enumerate(object_classes):
            # Magnitude pivot r_mag (luminosité apparente)
            r_pivot = np.random.uniform(14.0, 20.0)
            
            # Couleurs moyennes physiques selon la classe (cf. astrophysique standard du SDSS)
            if obj_class == 'ELLIPTICAL':
                # Très rouge (anciennes étoiles froides)
                u_g = 1.8 + np.random.normal(0, 0.15)
                g_r = 0.9 + np.random.normal(0, 0.08)
                r_i = 0.4 + np.random.normal(0, 0.05)
                i_z = 0.3 + np.random.normal(0, 0.05)
            elif obj_class == 'SPIRAL':
                # Intermédiaire (mélange jeune/vieux)
                u_g = 1.3 + np.random.normal(0, 0.20)
                g_r = 0.5 + np.random.normal(0, 0.10)
                r_i = 0.3 + np.random.normal(0, 0.06)
                i_z = 0.2 + np.random.normal(0, 0.06)
            elif obj_class == 'IRREGULAR':
                # Bleu (étoiles jeunes très chaudes)
                u_g = 0.8 + np.random.normal(0, 0.20)
                g_r = 0.2 + np.random.normal(0, 0.08)
                r_i = 0.1 + np.random.normal(0, 0.05)
                i_z = 0.1 + np.random.normal(0, 0.05)
            elif obj_class == 'QSO':
                # Très bleu ou très décalé selon le redshift
                u_g = 0.1 + np.random.normal(0, 0.30)
                g_r = 0.1 + np.random.normal(0, 0.15)
                r_i = 0.2 + np.random.normal(0, 0.10)
                i_z = 0.2 + np.random.normal(0, 0.10)
            else: # COMPACT_STAR
                # Étoiles : distribution Hertzsprung-Russell simplifiée
                u_g = np.random.uniform(0.5, 2.5)
                g_r = np.random.uniform(0.1, 1.2)
                r_i = g_r * 0.4
                i_z = g_r * 0.2
                
            r_mags[idx] = r_pivot
            g_mags[idx] = r_pivot + g_r
            u_mags[idx] = g_mags[idx] + u_g
            i_mags[idx] = r_pivot - r_i
            z_mags[idx] = i_mags[idx] - i_z
            
        # Création du DataFrame de métadonnées complet
        metadata = pd.DataFrame({
            'specobjid': np.arange(100000, 100000 + n_objects),
            'ra': np.random.uniform(0, 360, n_objects),
            'dec': np.random.uniform(-10, 70, n_objects),
            'class': object_classes,
            'subclass': ['NONE'] * n_objects,
            'z': redshifts,
            'u': u_mags,
            'g': g_mags,
            'r': r_mags,
            'i': i_mags,
            'z_mag': z_mags,
            'e_b_v': e_b_v
        })
        
        # Sauvegarde
        metadata.to_csv(os.path.join(self.cache_dir, "sdss_metadata.csv"), index=False)
        np.save(os.path.join(self.cache_dir, "sdss_wavelengths.npy"), wavelengths)
        np.save(os.path.join(self.cache_dir, "sdss_spectra.npy"), spectra)
        
        print("Génération terminée et sauvegardée dans le dossier cache `data/`.")
        return metadata, wavelengths, spectra
        
    def load_cached_data(self):
        """
        Charge les données du cache local s'il existe.
        """
        meta_path = os.path.join(self.cache_dir, "sdss_metadata.csv")
        wave_path = os.path.join(self.cache_dir, "sdss_wavelengths.npy")
        spec_path = os.path.join(self.cache_dir, "sdss_spectra.npy")
        
        if os.path.exists(meta_path) and os.path.exists(wave_path) and os.path.exists(spec_path):
            df = pd.read_csv(meta_path)
            wavelengths = np.load(wave_path)
            spectra = np.load(spec_path)
            return df, wavelengths, spectra
        return None
