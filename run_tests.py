import os
import sys

# Ajouter la racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_math import (
    test_randomized_svd_halko_properties, 
    test_sparse_graph_laplacian, 
    test_astrophysics_corrections, 
    test_quadrature_precision
)

def run_all_tests():
    print("=" * 70)
    print("    EXÉCUTION DU BANC DE TESTS ARCHITECTURAUX & SCIENTIFIQUES (V2)   ")
    print("=" * 70)
    
    tests = [
        ("Orthogonalité & Propriétés de la SVD Randomisée de Halko", test_randomized_svd_halko_properties),
        ("Graphe creux k-NN, Laplacien creux & Solveur ARPACK (eigsh)", test_sparse_graph_laplacian),
        ("Corrections Physiques : Doppler (Redshift) & Extinction galactique CCM", test_astrophysics_corrections),
        ("Précision et convergence de la Quadrature Numérique (ordre 2)", test_quadrature_precision)
    ]
    
    passed_count = 0
    
    for name, test_func in tests:
        print(f"[-] Running: {name} ...", end="", flush=True)
        try:
            test_func()
            print(" [PASS] [OK]")
            passed_count += 1
        except Exception as e:
            print(" [FAIL] [FAILED]")
            print(f"   Erreur détaillée : {e}")
            
    print("=" * 70)
    print(f"RÉSULTAT : {passed_count}/{len(tests)} tests physiques et mathématiques validés avec succès.")
    print("=" * 70)
    
    if passed_count == len(tests):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
