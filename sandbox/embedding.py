"""Feature embeddings used by the choroid clustering experiments."""

import numpy as np
from sklearn.decomposition import PCA


def _as_templates(templates):
    templates = np.asarray(templates, dtype=float)
    if templates.ndim != 2:
        raise ValueError("templates must be a 2-D array")
    if templates.shape[0] == 0 or templates.shape[1] < 3:
        raise ValueError("templates must contain at least one three-sample signal")
    if not np.all(np.isfinite(templates)):
        raise ValueError("templates must contain only finite values")
    return templates


def _selected_harmonics(template_length, n_harmonics, harmonics):
    max_harmonic = template_length // 2
    if harmonics is None:
        if not isinstance(n_harmonics, (int, np.integer)) or n_harmonics < 1:
            raise ValueError("n_harmonics must be a positive integer")
        harmonics = range(1, n_harmonics + 1)

    harmonics = tuple(harmonics)
    if not harmonics:
        raise ValueError("at least one harmonic must be selected")
    if any(
        not isinstance(k, (int, np.integer)) or k < 1 or k > max_harmonic
        for k in harmonics
    ):
        raise ValueError(
            f"harmonics must be integers between 1 and {max_harmonic}"
        )
    return harmonics


def _normalized_spectrum(template):
    centered = np.asarray(template, dtype=float) - np.mean(template)
    spectrum = np.fft.rfft(centered)
    fundamental_amplitude = np.abs(spectrum[1])
    tolerance = np.finfo(float).eps * max(1.0, np.linalg.norm(centered))
    if fundamental_amplitude <= tolerance:
        raise ValueError("cannot normalize a template with no first harmonic")
    return spectrum, fundamental_amplitude


def harmonic_features(template, n_harmonics=2, harmonics=None):
    """Encode selected Fourier harmonics as phase and relative amplitude."""
    template = _as_templates([template])[0]
    harmonics = _selected_harmonics(len(template), n_harmonics, harmonics)
    spectrum, fundamental_amplitude = _normalized_spectrum(template)

    features = []
    for k in harmonics:
        coefficient = spectrum[k]
        features.extend(
            (
                np.cos(np.angle(coefficient)),
                np.sin(np.angle(coefficient)),
                np.abs(coefficient) / fundamental_amplitude,
            )
        )
    return np.asarray(features)


def harmonic_embedding(templates, n_harmonics=2, harmonics=None):
    templates = _as_templates(templates)
    return np.asarray(
        [
            harmonic_features(
                template,
                n_harmonics=n_harmonics,
                harmonics=harmonics,
            )
            for template in templates
        ]
    )


def complex_fourier(template, n_harmonics=2, harmonics=None):
    """Encode selected complex Fourier coefficients relative to H1 amplitude."""
    template = _as_templates([template])[0]
    harmonics = _selected_harmonics(len(template), n_harmonics, harmonics)
    spectrum, fundamental_amplitude = _normalized_spectrum(template)

    embedding = []
    for k in harmonics:
        coefficient = spectrum[k] / fundamental_amplitude
        embedding.extend((coefficient.real, coefficient.imag))
    return np.asarray(embedding)


def complex_fourier_embedding(templates, n_harmonics=2, harmonics=None):
    templates = _as_templates(templates)
    return np.asarray(
        [
            complex_fourier(
                template,
                n_harmonics=n_harmonics,
                harmonics=harmonics,
            )
            for template in templates
        ]
    )


def PCA_embedding(templates, n_components=3, gradient=False):
    """Apply PCA to independently standardized cardiac-cycle shapes."""
    templates = _as_templates(templates)
    if gradient:
        templates = np.gradient(templates, axis=1)

    centered = templates - np.mean(templates, axis=1, keepdims=True)
    scales = np.std(centered, axis=1, keepdims=True)
    tolerance = np.finfo(float).eps * np.maximum(
        1.0,
        np.max(np.abs(centered), axis=1, keepdims=True),
    )
    if np.any(scales <= tolerance):
        raise ValueError("PCA embedding requires non-constant templates")
    standardized = centered / scales

    max_components = min(standardized.shape)
    if (
        not isinstance(n_components, (int, np.integer))
        or n_components < 1
        or n_components > max_components
    ):
        raise ValueError(f"n_components must be between 1 and {max_components}")
    return PCA(n_components=n_components).fit_transform(standardized)


def autocorrelate(template, n_lags=10):
    """Return positive-lag autocorrelation coefficients normalized at lag zero."""
    template = _as_templates([template])[0]
    if not isinstance(n_lags, (int, np.integer)) or not 1 <= n_lags < len(template):
        raise ValueError("n_lags must be between 1 and template length minus one")

    centered = template - np.mean(template)
    energy = np.dot(centered, centered)
    tolerance = np.finfo(float).eps * max(1.0, energy)
    if energy <= tolerance:
        raise ValueError("autocorrelation requires a non-constant template")

    full_correlation = np.correlate(centered, centered, mode="full")
    midpoint = len(full_correlation) // 2
    return full_correlation[midpoint + 1 : midpoint + 1 + n_lags] / energy


def autocorrelation_embedding(templates, n_lags=10, gradient=False):
    templates = _as_templates(templates)
    if gradient:
        templates = np.gradient(templates, axis=1)
    return np.asarray(
        [autocorrelate(template, n_lags=n_lags) for template in templates]
    )
