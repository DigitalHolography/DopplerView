from holosegment.pipeline.step import BaseStep, NestedStep
from holosegment.segmentation import pulse_analysis

class PulseAnalysisStep(NestedStep):
    name = "pulse_analysis"

    def __init__(self):
        self.substeps = [
            PreArteryMaskStep(),
            ComputeTemporalCuesStep()
        ]
        super().__init__()
            
class PreArteryMaskStep(BaseStep):
    requires = {"M0_ff_video", "retinal_vessel_mask", "optic_disc_center"}
    produces = {"pre_artery_mask"}
    name = "pre_artery_mask"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"]}

    def run(self, ctx):
        video = ctx.cache["M0_ff_video"]
        vessel_mask = ctx.cache["retinal_vessel_mask"]

        sampling_frequency = ctx.holodoppler_config["fs"]

        pre_artery_mask, pre_vein_mask = pulse_analysis.compute_pre_artery_mask(video, vessel_mask, ctx.cache["optic_disc_center"], sampling_frequency, ctx.output_manager)
        ctx.cache["pre_artery_mask"] = pre_artery_mask
        ctx.cache["pre_vein_mask"] = pre_vein_mask

class ComputeTemporalCuesStep(BaseStep):
    requires = {"M0_ff_video", "pre_artery_mask", "choroidal_vessel_mask"}
    produces = {"correlation", "diasys_image", "retinal_arterial_pulse", "choroidal_pulse"}
    name = "temporal_cues"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"],
                "batch_stride": ctx.holodoppler_config["batch_stride"]}

    def run(self, ctx):
        video = ctx.require("M0_ff_video")
        pre_artery_mask = ctx.require("pre_artery_mask")
        choroidal_vessel_mask = ctx.require("choroidal_vessel_mask")

        correlation, pulse = pulse_analysis.compute_correlation(video, pre_artery_mask)
        correlation_choroidal, pulse_choroidal = pulse_analysis.compute_correlation(video, choroidal_vessel_mask)
        ctx.set("correlation", correlation)

        sampling_frequency = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        diasys, M0_Systole_img, M0_Diastole_img, fullPulse = pulse_analysis.compute_diasys_image(video, pre_artery_mask, sampling_frequency=sampling_frequency, stride=stride)
        choroid_diasys, choroid_systole, choroid_diastole, choroid_fullPulse = pulse_analysis.compute_diasys_image(video, choroidal_vessel_mask, sampling_frequency=sampling_frequency, stride=stride)

        ctx.set("retinal_arterial_pulse", fullPulse)
        ctx.set("choroidal_pulse", choroid_fullPulse)
        ctx.set("diasys_image", diasys)
