from holosegment.steps.step import BaseStep
import numpy as np

class AVSegmentationStep(BaseStep):
    requires = ["M0_ff_video", "M0_ff_image", "temporal_cues"]
    produces = ["artery_mask", "vein_mask"]
    name = "av_segmentation"

    def deep_segmentation(self, ctx):
        # model_name = ctx.config["models"]["av"]
        model_name = "nnwnet_av_corr_diasys"
        model = ctx.get_model(model_name)
        M0 = ctx.require("M0_ff_image")
        cues = ctx.require("temporal_cues")

        print(M0.shape, cues["correlation"].shape, cues["diasys"].shape)

        
        input = np.stack([M0, cues["correlation"], cues["diasys"]], axis=0)  # shape (3, H, W)

        print(input.shape)

        mask = model.predict(input)
        mask = np.squeeze(mask)  # Remove channel dimension if present

        if model.spec.output_activation == "argmax":
            return np.where((mask==1) | (mask==3), 1, 0), np.where((mask==2) | (mask==3), 1, 0)
        
        return mask[0], mask[1]

    def handmade_segmentation(self, ctx):
        raise NotImplementedError("Handmade artery vein segmentation not implemented yet.")

    def run(self, ctx):
        if ctx.config.get("AVSegmentationMethod", "AI") == "AI":
            print("Using deep segmentation model for artery vein segmentation.")
            ctx.cache["artery_mask"], ctx.cache["vein_mask"] = self.deep_segmentation(ctx)
            
        else:
            print("Use hand-made heuristics for artery vein segmentation.")
            ctx.cache["artery_mask"], ctx.cache["vein_mask"] = self.handmade_segmentation(ctx)
        
        ctx.output_manager.save(self.name, "artery_mask", ctx.cache["artery_mask"], "png")
        ctx.output_manager.save(self.name, "vein_mask", ctx.cache["vein_mask"], "png")
