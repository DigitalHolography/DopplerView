from holosegment.pipeline.step import BaseStep, NestedStep
from holosegment.segmentation import pulse_analysis

class PulseAnalysisStep(NestedStep):
    requires = ["M0_ff_video", "vessel_mask"]
    produces = ["correlation", "diasys_image"]
    name = "pulse_analysis"

    def __init__(self):
        self.substeps = [
            PreArteryMaskStep(),
            ComputeTemporalCuesStep()
        ]

    def run(self, ctx):
        for step in self.substeps:
            step.run(ctx)
            
class PreArteryMaskStep(BaseStep):
    requires = ["M0_ff_video", "vessel_mask", "optic_disc_center"]
    produces = ["pre_artery_mask"]
    name = "pre_artery_mask"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"]}

    def run(self, ctx):
        video = ctx.cache["M0_ff_video"]
        vessel_mask = ctx.cache["vessel_mask"]

        sampling_frequency = ctx.holodoppler_config["fs"]

        pre_artery_mask, pre_vein_mask = pulse_analysis.compute_pre_artery_mask(video, vessel_mask, ctx.cache["optic_disc_center"], sampling_frequency, ctx.output_manager)
        ctx.output_manager.save(self.name, "pre_artery_mask", pre_artery_mask, "png")
        ctx.cache["pre_artery_mask"] = pre_artery_mask
        ctx.output_manager.save(self.name, "pre_vein_mask", pre_vein_mask, "png")
        ctx.cache["pre_vein_mask"] = pre_vein_mask

class ComputeTemporalCuesStep(BaseStep):
    requires = ["M0_ff_video", "pre_artery_mask"]
    produces = ["correlation", "diasys_image"]

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"],
                "batch_stride": ctx.holodoppler_config["batch_stride"]}

    def run(self, ctx):
        video = ctx.cache["M0_ff_video"]
        pre_artery_mask = ctx.cache["pre_artery_mask"]

        vessel_mask = ctx.cache["vessel_mask"]

        correlation = pulse_analysis.compute_correlation(video, vessel_mask)
        ctx.set("correlation", correlation)
        ctx.output_manager.save(self.name, "correlation_map", correlation, "png")

        sampling_frequency = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        diasys, M0_Systole_img, M0_Diastole_img, fullPulse = pulse_analysis.compute_diasys_image(video, vessel_mask, sampling_frequency=sampling_frequency, stride=stride)
        ctx.output_manager.save(self.name, "diasys_image", diasys, "png")
        ctx.output_manager.save(self.name, "M0_Systole_img", M0_Systole_img, "png")
        ctx.output_manager.save(self.name, "M0_Diastole_img", M0_Diastole_img, "png")
        ctx.output_manager.save_plot(self.name, "fullPulse", fullPulse, title = "Full Pulse Analysis (mean intensity over time)")

        ctx.set("diasys_image", diasys)
