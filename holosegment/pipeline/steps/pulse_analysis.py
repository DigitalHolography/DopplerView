from unittest import signals

from holosegment.pipeline.step import BaseStep, NestedStep
from holosegment.segmentation import process_masks, pulse_analysis
import numpy as np

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
    produces = {"labeled_vessels", "pre_artery_mask", "branch_signals"}
    name = "pre_artery_mask"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"]}

    def run(self, ctx):
        video = ctx.cache["M0_ff_video"]
        vessel_mask = ctx.cache["retinal_vessel_mask"]
        optic_disc_center = ctx.cache["optic_disc_center"]

        fs = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        sampling_frequency = pulse_analysis.get_effective_sampling_freqency(fs, stride)

        # --- Step 1: Separate mask into branches ---
        labeled_vessels, _ = process_masks.get_labeled_vesselness(vessel_mask, *optic_disc_center)
        ctx.set("labeled_vessels", labeled_vessels)

        # --- Step 2: Compute mean temporal signal for each branch ---
        signals = pulse_analysis.get_filtered_branch_signals(video, labeled_vessels, sampling_frequency)
        for i in range(1, labeled_vessels.max() + 1):
            ctx.output_manager.debug("pulse_analysis", f"branch_{i}_signal", signals[i - 1, :], "signal")
        signals_n = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)
        ctx.cache["branch_signals"] = signals_n

        # --- Step 3: Select regular peaks to classify arteries vs veins ---
        pre_artery_mask, pre_vein_mask = pulse_analysis.compute_pre_masks(signals_n, labeled_vessels, sampling_frequency)
        ctx.cache["pre_artery_mask"] = pre_artery_mask
        ctx.cache["pre_vein_mask"] = pre_vein_mask

class ComputeTemporalCuesStep(BaseStep):
    requires = {"M0_ff_video", "pre_artery_mask", "choroidal_vessel_mask"}
    produces = {"correlation", "diasys_image", "retinal_arterial_pulse", "choroidal_pulse", "retinal_arterial_pulse_filtered", "choroidal_pulse_filtered"}
    name = "temporal_cues"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"],
                "batch_stride": ctx.holodoppler_config["batch_stride"]}

    def run(self, ctx):
        video = ctx.require("M0_ff_video")
        pre_artery_mask = ctx.require("pre_artery_mask")
        pre_vein_mask = ctx.require("pre_vein_mask")
        choroidal_vessel_mask = ctx.require("choroidal_vessel_mask")

        # --- Get pulses from masks ---

        arterial_pulse = pulse_analysis.get_pulse_from_mask(video, pre_artery_mask)
        venous_pulse = pulse_analysis.get_pulse_from_mask(video, pre_vein_mask)
        choroidal_pulse = pulse_analysis.get_pulse_from_mask(video, choroidal_vessel_mask)

        # --- Filter pulses to remove high frequency noise ---

        fs = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        sampling_frequency = pulse_analysis.get_effective_sampling_freqency(fs, stride)

        arterial_pulse_filtered = pulse_analysis.get_filtered_pulse(arterial_pulse, sampling_frequency)
        venous_pulse_filtered = pulse_analysis.get_filtered_pulse(venous_pulse, sampling_frequency)
        choroidal_pulse_filtered = pulse_analysis.get_filtered_pulse(choroidal_pulse, sampling_frequency)

        # --- Compute correlation map with filtered pulses ---

        correlation_artery = pulse_analysis.compute_correlation(video, arterial_pulse_filtered)
        # correlation_vein = pulse_analysis.compute_correlation(video, venous_pulse_filtered)
        ctx.set("correlation", correlation_artery)
        # ctx.set("correlation_vein", correlation_vein)

        # --- Accumulate frames at the systolic and diastolic peaks of the filtered pulses ---

        diasys, M0_Systole_img, M0_Diastole_img = pulse_analysis.compute_diasys_image(video, arterial_pulse_filtered, sampling_frequency)

        ctx.set("pre_arterial_pulse", arterial_pulse)
        ctx.set("choroidal_pulse", choroidal_pulse)
        ctx.set("pre_arterial_pulse_filtered", arterial_pulse_filtered)
        ctx.set("choroidal_pulse_filtered", choroidal_pulse_filtered)
        ctx.set("diasys_image", diasys)
