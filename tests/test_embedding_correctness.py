import numpy as np
import pytest

from sandbox import embedding


def _pairwise_distances(values):
    return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)


def test_pca_embedding_standardizes_each_template_independently():
    x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    templates = np.asarray(
        [
            np.sin(x),
            np.sin(x + 0.4) + 0.2 * np.sin(2 * x),
            np.sin(x + 0.9) - 0.1 * np.cos(3 * x),
        ]
    )
    independently_rescaled = templates * np.asarray([[2.0], [7.0], [0.3]])
    independently_rescaled += np.asarray([[10.0], [-4.0], [100.0]])

    original = embedding.PCA_embedding(templates, n_components=2)
    transformed = embedding.PCA_embedding(independently_rescaled, n_components=2)

    np.testing.assert_allclose(
        _pairwise_distances(original),
        _pairwise_distances(transformed),
        atol=1e-10,
    )


def test_autocorrelation_is_offset_and_scale_invariant():
    x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    template = np.sin(x) + 0.25 * np.sin(2 * x)

    expected = embedding.autocorrelate(template, n_lags=8)
    actual = embedding.autocorrelate(5.0 * template + 17.0, n_lags=8)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.all(np.abs(actual) <= 1.0)


@pytest.mark.parametrize(
    "function, kwargs",
    [
        (embedding.harmonic_features, {}),
        (embedding.complex_fourier, {}),
        (embedding.autocorrelate, {}),
    ],
)
def test_degenerate_templates_are_rejected(function, kwargs):
    with pytest.raises(ValueError):
        function(np.ones(16), **kwargs)


def test_fourier_embeddings_reject_unavailable_harmonics():
    with pytest.raises(ValueError, match="harmonics"):
        embedding.complex_fourier(np.arange(8), harmonics=[5])


def test_embedding_rejects_nonfinite_input():
    templates = np.asarray([[0.0, 1.0, np.nan], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        embedding.PCA_embedding(templates, n_components=1)
