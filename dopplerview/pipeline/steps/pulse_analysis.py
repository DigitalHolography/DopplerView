from unittest import signals

from dopplerview.pipeline.step import BaseStep, NestedStep
from dopplerview.segmentation import process_masks, pulse_analysis, signal_processing
from dopplerview.utils.parallelization_utils import run_in_parallel
from functools import partial

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
    produces = {"labeled_vessels", "pre_artery_mask", "branch_signals", "corrected_signals", "pre_vein_mask"}
    name = "pre_artery_mask"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"]}

    def run(self, ctx):
        video = ctx.cache["M0_ff_video"]
        vessel_mask = ctx.cache["retinal_vessel_mask"]
        optic_disc_center = ctx.cache["optic_disc_center"]

        fs = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)

        # --- Step 1: Separate mask into branches ---
        labeled_vessels, _ = process_masks.get_labeled_vesselness(vessel_mask, *optic_disc_center)
        ctx.set("labeled_vessels", labeled_vessels)

        # --- Step 2: Compute mean temporal signal for each branch ---
        signals = pulse_analysis.get_filtered_branch_signals(video, labeled_vessels, sampling_frequency)
        signals_n = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)
        ctx.cache["branch_signals"] = signals_n

        # --- Step 3: Correct signals by aligning with median heartbeat ---
        beat_period = pulse_analysis.compute_idx0(signals_n, sampling_frequency)
        corrected_signals = np.zeros_like(signals_n)
        func = partial(pulse_analysis.correct_branch_signal_with_heartbeat, beat_period=beat_period, k=10)
        corrected_signals = run_in_parallel(func, signals_n, n_jobs=-1, chunking=False)
        # for i, signal in enumerate(signals_n):
        #     corrected_signals[i, :] = pulse_analysis.correct_branch_signal_with_heartbeat(signal, beat_period, k=10)
        ctx.cache["corrected_signals"] = corrected_signals
        for i in range(1, labeled_vessels.max() + 1):
            ctx.output_manager.output("pulse_analysis", f"branch_{i}_corrected", (signals_n[i - 1, :], corrected_signals[i - 1, :]), "signal", options={"multiple_signals": True, "legend": ["Original Signal", "Corrected Signal"]})

        # --- Step 4: Pre-classify arteries and veins using systolic gradient ---
        pre_artery_mask, pre_vein_mask = pulse_analysis.compute_pre_masks(corrected_signals, labeled_vessels, sampling_frequency)
        ctx.cache["pre_artery_mask"] = pre_artery_mask
        ctx.cache["pre_vein_mask"] = pre_vein_mask

class ComputeTemporalCuesStep(BaseStep):
    requires = {"M0_ff_video", "pre_artery_mask", "choroidal_vessel_mask"}
    produces = {"correlation", "diasys_image", "pre_arterial_pulse", "choroidal_pulse", "pre_arterial_pulse_filtered", "choroidal_pulse_filtered", "pre_arterial_pulse_interpolated", "pre_venous_pulse", "pre_venous_pulse_filtered"}
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

        arterial_pulse = signal_processing.get_pulse_from_mask(video, pre_artery_mask)
        venous_pulse = signal_processing.get_pulse_from_mask(video, pre_vein_mask)
        choroidal_pulse = signal_processing.get_pulse_from_mask(video, choroidal_vessel_mask)

        # --- Filter pulses to remove high frequency noise ---

        fs = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)

        arterial_pulse_filtered = signal_processing.get_filtered_pulse(arterial_pulse, sampling_frequency)
        venous_pulse_filtered = signal_processing.get_filtered_pulse(venous_pulse, sampling_frequency)
        choroidal_pulse_filtered = signal_processing.get_filtered_pulse(choroidal_pulse, sampling_frequency)

        # --- Interpolate outlier frames using the filtered signal ---

        video_cleaned, arterial_pulse_interpolated = signal_processing.interpolate_outliers(video, arterial_pulse, pre_artery_mask, sampling_frequency=sampling_frequency)
        # ctx.output_manager.output("pulse_analysis", "video_cleaned", video_cleaned, "video")

        # --- Compute correlation map with filtered pulses ---

        correlation_artery = signal_processing.compute_correlation(video_cleaned, arterial_pulse_interpolated)
        # correlation_vein = pulse_analysis.compute_correlation(video, venous_pulse_filtered)
        ctx.set("correlation", correlation_artery)
        # ctx.set("correlation_vein", correlation_vein)

        # --- Accumulate frames at the systolic and diastolic peaks of the filtered pulses ---

        diasys, M0_Systole_img, M0_Diastole_img = pulse_analysis.compute_diasys_image(video_cleaned, arterial_pulse_interpolated, sampling_frequency)

        ctx.set("pre_arterial_pulse", arterial_pulse)
        ctx.set("pre_venous_pulse", venous_pulse)
        ctx.set("choroidal_pulse", choroidal_pulse)
        ctx.set("pre_arterial_pulse_filtered", arterial_pulse_filtered)
        ctx.set("pre_venous_pulse_filtered", venous_pulse_filtered)
        ctx.set("pre_arterial_pulse_interpolated", arterial_pulse_interpolated)
        ctx.set("choroidal_pulse_filtered", choroidal_pulse_filtered)
        ctx.set("diasys_image", diasys)
