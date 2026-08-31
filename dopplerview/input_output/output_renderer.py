import imageio
from dopplerview.utils.image_utils import normalize_to_uint8, save_numpy_as_avi, save_labeled_branches, normalize_image, lab_duo_image
from dopplerview.utils.matplotlib_backend import new_agg_figure, serialized_render
from matplotlib.patches import Rectangle
import numpy as np

class OutputRenderer:
    def required_keys(self, key):
        return {key}

    def render(self, key, ctx, path, options=None):
        raise NotImplementedError
    
class ImageRenderer(OutputRenderer):
    def render(self, key, ctx, path, options=None):
        if options and options.get("blue_gray_red"):
            M0_ff_image = normalize_image(options.get("M0_ff_image"))
            img = lab_duo_image(M0_ff_image, ctx.get(key))
            imageio.imwrite(path, normalize_to_uint8(img))

            # plt.imsave(path, normalize_to_uint8(img), cmap=cmap, vmin=0, vmax=255)
        else:
            imageio.imwrite(path, normalize_to_uint8(ctx.get(key)))

class MaskRenderer(OutputRenderer):
    def render(self, key, ctx, path, options=None):
        """Render a binary mask with foreground pixels visibly white.
        """
        mask = np.asarray(ctx.get(key))
        imageio.imwrite(path, (mask > 0).astype(np.uint8) * 255)

class SignalRenderer(OutputRenderer):
    @serialized_render
    def render(self, key, ctx, path, options=None):
        fig = new_agg_figure()
        ax = fig.subplots()
        ax.set_title(key)

        scatter_indices = None
        if options and options.get("scatter"):
            scatter_indices = ctx.get(key)[-1] # Only scatter the last signal if multiple_signals is True

        if options and (options.get("multiple_signals") or options.get("scatter")):
            legend = options.get("legend", [])
            for i, signal in enumerate(ctx.get(key)):
                if scatter_indices is not None:
                    if i == len(ctx.get(key)) - 1:  # Skip las iteration if scatter is True
                        continue
                    ax.scatter(scatter_indices, signal[scatter_indices], label=f"{legend[i]} Peaks" if i < len(legend) else "Peaks", s=50)
                ax.plot(signal, label=legend[i] if i < len(legend) else "")
            if legend:
                ax.legend()
        else:
            ax.plot(ctx.get(key))
        fig.savefig(path)
        fig.clear()

class VideoRenderer(OutputRenderer):
    def render(self, key, ctx, path, options=None):
        save_numpy_as_avi(ctx.get(key), path.with_suffix(".avi"))

class OpticDiscRenderer(OutputRenderer):
    def required_keys(self, key):
        return {
            key,
            "M0_ff_image",
            "optic_disc_center",
            "optic_disc_width",
            "optic_disc_height",
        }

    @serialized_render
    def render(self, key, ctx, path, options=None):
        image = ctx.get("M0_ff_image")
        center = ctx.get("optic_disc_center")
        diameter_x = ctx.get("optic_disc_width")
        diameter_y = ctx.get("optic_disc_height")

        x_center, y_center = center

        a = diameter_x / 2
        b = diameter_y / 2

        angle = np.linspace(0, 2*np.pi, 360)

        x = x_center + a*np.cos(angle)
        y = y_center + b*np.sin(angle)

        fig = new_agg_figure(figsize=(6, 6))
        ax = fig.subplots()
        ax.imshow(image, cmap="gray")

        ax.plot(x, y, "r")

        ax.add_patch(
            Rectangle(
                (x_center-a, y_center-b),
                diameter_x,
                diameter_y,
                fill=False,
                edgecolor="lime"
            )
        )

        fig.savefig(path)
        fig.clear()

class LabeledMaskRenderer(OutputRenderer):
    @serialized_render
    def render(self, key, ctx, path, options=None):
        save_labeled_branches(ctx.get(key), path)
