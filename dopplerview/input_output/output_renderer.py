import imageio
from dopplerview.utils.image_utils import normalize_to_uint8, save_numpy_as_avi, save_labeled_branches
import matplotlib.pyplot as plt
import numpy as np

class DebugRenderer:
    def render(self, key, cache, path, options=None):
        raise NotImplementedError
    
class ImageRenderer(DebugRenderer):
    def render(self, key, cache, path, options=None):
        imageio.imwrite(path, normalize_to_uint8(cache.get(key)))

class SignalRenderer(DebugRenderer):
    def render(self, key, cache, path, options=None):
        plt.figure()
        plt.title(key)
        if options and options.get("multiple_signals"):
            legend = options.get("legend", [])
            for i, signal in enumerate(cache.get(key)):
                plt.plot(signal, label=legend[i] if i < len(legend) else "")
            if legend:
                plt.legend()
        else:
            plt.plot(cache.get(key))
        plt.savefig(path)
        plt.close()

class VideoRenderer(DebugRenderer):
    def render(self, key, cache, path, options=None):
        save_numpy_as_avi(cache.get(key), path.with_suffix(".avi"))

class OpticDiscRenderer(DebugRenderer):
    def render(self, key, cache, path, options=None):
        image = cache.get("M0_ff_image")
        center = cache.get("optic_disc_center")
        axes = cache.get("optic_disc_axes")

        x_center, y_center = center
        diameter_x, diameter_y = axes

        a = diameter_x / 2
        b = diameter_y / 2

        angle = np.linspace(0, 2*np.pi, 360)

        x = x_center + a*np.cos(angle)
        y = y_center + b*np.sin(angle)

        plt.figure(figsize=(6,6))
        plt.imshow(image, cmap="gray")

        plt.plot(x, y, "r")

        plt.gca().add_patch(
            plt.Rectangle(
                (x_center-a, y_center-b),
                diameter_x,
                diameter_y,
                fill=False,
                edgecolor="lime"
            )
        )

        plt.savefig(path)
        plt.close()

class LabeledMaskRenderer(DebugRenderer):
    def render(self, key, cache, path, options=None):
        save_labeled_branches(cache.get(key), path)
        plt.close()