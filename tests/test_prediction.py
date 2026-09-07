"""
Tests unitaires pour le projet :
https://github.com/AlexandruEmil/Data-Science-API-FastAPI-Docker

Emplacement recommandé dans le dépôt cloné :
    tests/unit/test_prediction.py

Objectif pédagogique :
- Tester uniquement la fonction predict(), sans lancer FastAPI.
- Couvrir les cas nominaux, limites, invalides et exceptionnels.
- Éviter une couverture artificielle basée seulement sur des cas répétitifs.
"""

import pytest

from app.utils import predict


# -----------------------------------------------------------------------------
# Cas nominaux : entrées valides et représentatives
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "features, expected",
    [
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        ([5.0], [10.0]),
        ([1.5, 2.5], [4.0, 5.0]),
    ],
)

def test_predict_nominal_cases(features, expected):
    """La fonction doit retourner une prédiction conforme à la règle y = 2x."""
    result = predict(features)
    assert result == pytest.approx(expected)



# -----------------------------------------------------------------------------
# Cas limites : valeurs particulières aux frontières du domaine
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "features, expected",
    [
        ([0.0], [0.0]),
        ([-1.0], [-2.0]),
        ([-1000.0], [-2000.0]),
        ([1_000_000.0], [2_000_000.0]),
        ([0.0, -1.0, 1_000_000.0], [0.0, -2.0, 2_000_000.0]),
    ],
)
def test_predict_boundary_cases(features, expected):
    """La fonction doit gérer correctement les valeurs limites ou particulières."""
    result = predict(features)

    assert result == pytest.approx(expected)


# -----------------------------------------------------------------------------
# Cas exceptionnel : entrée techniquement sous forme de liste, mais inexploitable
# -----------------------------------------------------------------------------

def test_predict_empty_list_raises_exception():
    """
    Une liste vide ne contient aucun échantillon à prédire.
    Le modèle scikit-learn doit donc lever une exception.
    """
    with pytest.raises(ValueError):
        predict([])
        


# -----------------------------------------------------------------------------
# Cas invalides : données ne respectant pas les préconditions attendues
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_features",
    [
        None,
        "abc",
        {"feature1": 1.0},
        [1.0, "abc", 3.0],
        [1.0, None, 3.0],
    ],
)
def test_predict_invalid_inputs_raise_exception(invalid_features):
    """La fonction doit échouer de manière contrôlée avec des entrées invalides."""
    with pytest.raises((TypeError, ValueError)):
        predict(invalid_features)



# -----------------------------------------------------------------------------
# Propriétés générales : tests plus robustes qu'une simple valeur attendue
# -----------------------------------------------------------------------------

def test_predict_output_is_a_list():
    """La fonction doit retourner une liste Python."""
    result = predict([1.0, 2.0, 3.0])

    assert isinstance(result, list)


def test_predict_output_size_matches_input_size():
    """Le nombre de prédictions doit correspondre au nombre d'entrées."""
    features = [1.0, 2.0, 3.0, 4.0]

    result = predict(features)

    assert len(result) == len(features)



def test_predict_is_deterministic():
    """La fonction doit être rejouable : mêmes entrées, mêmes sorties."""
    features = [3.5, 1.2, 4.9]

    first_result = predict(features)
    second_result = predict(features)

    assert first_result == pytest.approx(second_result)


def test_predict_does_not_modify_input_list():
    """La fonction ne doit pas modifier la liste reçue en entrée."""
    features = [1.0, 2.0, 3.0]
    original_features = features.copy()

    predict(features)

    assert features == original_features