from dopplerview.pipeline.step import BaseStep, NestedStep
from dopplerview.segmentation import process_masks, pulse_analysis, signal_processing
import dopplerview.utils.image_utils as image_utils


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
        return {
            "sampling_freq": ctx.holodoppler_config["sampling_freq"],
            "batch_stride": ctx.holodoppler_config["batch_stride"],
            "NumberOfWorkers": ctx.dopplerview_config["NumberOfWorkers"]
            }

    def run(self, ctx):
        video = ctx.get("M0_ff_video")
        vessel_mask = ctx.get("retinal_vessel_mask")
        optic_disc_center = ctx.get("optic_disc_center")

        fs = ctx.holodoppler_config["sampling_freq"]
        stride = ctx.holodoppler_config["batch_stride"]
        self.logger.info(f"    - Camera sampling frequency: {fs} Hz, batch stride: {stride}")

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)

        self.logger.info(f"    - Effective sampling frequency after accounting for batch stride: {sampling_frequency:.2f} Hz")

        # --- Step 1: Separate mask into branches ---
        labeled_vessels, _ = process_masks.get_labeled_vesselness(vessel_mask, *optic_disc_center)
        ctx.set("labeled_vessels", labeled_vessels)

        # --- Step 2: Compute mean temporal signal for each branch ---
        # ctx.output_manager.output("pulse_analysis", "M0_ff_video", video, "video")
        signals = pulse_analysis.get_filtered_branch_signals(video, labeled_vessels, sampling_frequency)
        ctx.output_manager.output("pulse_analysis", "labeled_vessels", labeled_vessels, "labeled_mask")
        signals_n = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)
        ctx.set("branch_signals", signals_n)
        ctx.output_manager.save_h5("branch_signals", ctx)

        # # --- Step 3: Correct signals by aligning with median heartbeat ---
        # self.logger.info("    - Sampling frequency: {:.2f} Hz, Beat period: {:.2f} seconds".format(sampling_frequency, beat_period))
        # corrected_signals = np.zeros_like(signals_n)
        # func = partial(pulse_analysis.correct_branch_signal_with_heartbeat, beat_period=beat_period, k=5)
        # corrected_signals = run_in_parallel(func, signals_n, n_jobs=ctx.dopplerview_config["NumberOfWorkers"], chunking=False, task_name="branch signal correction")
        # # for i, signal in enumerate(signals_n):
        # #     corrected_signals[i, :] = pulse_analysis.correct_branch_signal_with_heartbeat(signal, beat_period, k=10)
        # ctx.set("corrected_signals", corrected_signals)

        # for i in range(1, labeled_vessels.max() + 1):
        #     ctx.output_manager.output("pulse_analysis", f"branch_{i}_corrected", (signals_n[i - 1, :], corrected_signals[i - 1, :]), "signal", options={"multiple_signals": True, "legend": ["Original Signal", "Corrected Signal"]})

        # --- Step 4: Pre-classify arteries and veins using systolic gradient ---
        pre_artery_mask, pre_vein_mask, labels, z = pulse_analysis.compute_pre_masks(signals_n, labeled_vessels, sampling_frequency)
        ctx.output_manager.save_clusterization("pulse_analysis", "pre_mask_clusterization", labels, z)
        ctx.output_manager.save_overlay("pulse_analysis", "av_overlay_pre_masks", ctx.require("M0_ff_image"), pre_artery_mask, pre_vein_mask)
        ctx.set("pre_artery_mask", pre_artery_mask)
        ctx.set("pre_vein_mask", pre_vein_mask)

class ComputeTemporalCuesStep(BaseStep):
    requires = {"M0_ff_video", "pre_artery_mask", "choroidal_vessel_mask"}
    produces = {"correlation", "diasys_image", "pre_arterial_pulse", "choroidal_pulse", "pre_arterial_pulse_filtered", "choroidal_pulse_filtered", "pre_arterial_pulse_cleaned", "pre_venous_pulse", "pre_venous_pulse_filtered", "M0_ff_image_cleaned", "beat_period", "systole_image", "diastole_image"}
    name = "temporal_cues"

    def _relevant_config(self, ctx):
        return {"sampling_freq": ctx.holodoppler_config["sampling_freq"],
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
        ctx.set("pre_arterial_pulse", arterial_pulse)
        ctx.set("pre_venous_pulse", venous_pulse)
        ctx.set("choroidal_pulse", choroidal_pulse)

        # --- Filter pulses to remove high frequency noise ---

        fs = ctx.holodoppler_config["sampling_freq"]
        stride = ctx.holodoppler_config["batch_stride"]

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)

        arterial_pulse_filtered = signal_processing.get_filtered_pulse(arterial_pulse, sampling_frequency)
        venous_pulse_filtered = signal_processing.get_filtered_pulse(venous_pulse, sampling_frequency)
        choroidal_pulse_filtered = signal_processing.get_filtered_pulse(choroidal_pulse, sampling_frequency)
        ctx.set("pre_arterial_pulse_filtered", arterial_pulse_filtered)
        ctx.set("pre_venous_pulse_filtered", venous_pulse_filtered)
        ctx.set("choroidal_pulse_filtered", choroidal_pulse_filtered)

        # --- Remove bad beats from arterial pulse by comparing to median beat pattern ---

        beat_period = pulse_analysis.compute_period(arterial_pulse_filtered, sampling_frequency)
        ctx.set("beat_period", beat_period)

        arterial_pulse_cleaned, video_cleaned, beat_signal, median_beat, peaks = pulse_analysis.remove_bad_beats(arterial_pulse_filtered, video, beat_period, threshold=0.8)
        ctx.set("pre_arterial_pulse_cleaned", arterial_pulse_cleaned)

        M0_ff_image_cleaned = image_utils.normalize_to_uint8(np.mean(video_cleaned, axis=0))
        ctx.set("M0_ff_image_cleaned", M0_ff_image_cleaned)

        self.logger.info(f"    - Removed {len(arterial_pulse_filtered) - len(arterial_pulse_cleaned)} frames due to low correlation with median arterial pulse beat pattern")
        ctx.output_manager.output("pulse_analysis", f"outlier removal", (arterial_pulse_filtered, beat_signal), "signal", options={"multiple_signals": True, "legend": ["Original Signal", "beat signal"]})
        ctx.output_manager.output("pulse_analysis", f"median beat", median_beat, "signal")
        ctx.output_manager.output("pulse_analysis", f"peaks", (arterial_pulse_filtered, peaks), "signal", options={"scatter": True})

        # --- Compute correlation map with filtered pulses ---

        correlation_artery = signal_processing.compute_correlation(video_cleaned, arterial_pulse_cleaned)
        ctx.set("correlation", correlation_artery)
        ctx.output_manager.output("pulse_analysis", f"correlation map RGB", correlation_artery, "image", options={"blue_gray_red": True, "M0_ff_image": M0_ff_image_cleaned})
        # correlation_vein = pulse_analysis.compute_correlation(video, venous_pulse_filtered)
        # ctx.set("correlation_vein", correlation_vein)

        # --- Accumulate frames at the systolic and diastolic peaks of the filtered pulses ---

        diasys, sysindexes, diasindexes, systole, diastole = pulse_analysis.compute_diasys_image(video_cleaned, arterial_pulse_cleaned, sampling_frequency)
        ctx.output_manager.output("pulse_analysis", f"diasys image RGB", diasys, "image", options={"blue_gray_red": True, "M0_ff_image": M0_ff_image_cleaned})
        ctx.output_manager.output("pulse_analysis", f"diasys plot", (arterial_pulse_cleaned, sysindexes), "signal", options={"scatter": True})

        ctx.set("diasys_image", diasys)
        ctx.set("systole_image", systole)
        ctx.set("diastole_image", diastole)



