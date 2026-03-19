from holosegment.pipeline.step import BaseStep
import numpy as np

class AVSegmentationStep(BaseStep):
    requires = {"M0_ff_video", "M0_ff_image", "correlation", "diasys_image"}
    produces = {"retinal_artery_mask", "retinal_vein_mask"}
    name = "retinal_artery_vein_segmentation"

    def _relevant_config(self, ctx):
        params = ctx.eyeflow_config["Mask"]
        return { "AVSegmentationMethod": params.get("AVSegmentationMethod", "AI"),
                    "av_segmentation_model": ctx.get_current_model_for_task(self.name)
        }

    def deep_segmentation(self, ctx):
        # model_name = ctx.eyeflow_config["models"]["av"]
        model = ctx.get_current_model_for_task(self.name)

        input = model.prepare_input(ctx)

        mask = model.predict(input)
        mask = np.squeeze(mask)  # Remove channel dimension if present

        if model.spec.output_activation == "argmax":
            return np.where((mask==1) | (mask==3), 1, 0), np.where((mask==2) | (mask==3), 1, 0)
        
        return mask[0], mask[1]

    def handmade_segmentation(self, ctx):
        raise NotImplementedError("Handmade artery vein segmentation not implemented yet.")

    def run(self, ctx):
        if ctx.eyeflow_config.get("AVSegmentationMethod", "AI") == "AI":
            print("    - Use deep segmentation model for artery vein segmentation.")
            ctx.cache["retinal_artery_mask"], ctx.cache["retinal_vein_mask"] = self.deep_segmentation(ctx)
            
        else:
            print("    - Use hand-made heuristics for artery vein segmentation.")
            ctx.cache["retinal_artery_mask"], ctx.cache["retinal_vein_mask"] = self.handmade_segmentation(ctx)
        