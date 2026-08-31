from dopplerview.pipeline.step import BaseStep, NestedStep
from dopplerview.segmentation import process_masks, pulse_analysis, signal_processing
import dopplerview.utils.image_utils as image_utils
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
    requires = {"M0_ff_video", "M0_ff_image", "retinal_vessel_mask", "optic_disc_center"}
    produces = {
        "labeled_vessels",
        "pre_artery_mask",
        "branch_signals",
        "pre_vein_mask",
        "pre_artery_mask_clustering",
        "pre_vein_mask_clustering",
        "pre_artery_mask_gradient",
        "pre_vein_mask_gradient",
    }
    name = "pre_artery_mask"

    def _relevant_config(self, ctx):
        return {
            "sampling_freq": ctx.holodoppler_config.get("sampling_freq", 37037),
            "batch_stride": ctx.holodoppler_config.get("batch_stride", 256),
            "PreMaskMethod": ctx.dopplerview_config.get("Mask").get("PreMaskMethod", "clustering"),
            "CorrectBranchSignals": ctx.dopplerview_config.get("Mask").get("CorrectBranchSignals", True),
            }

    def run(self, ctx):
        video = ctx.get("M0_ff_video")
        vessel_mask = ctx.get("retinal_vessel_mask")
        optic_disc_center = ctx.get("optic_disc_center")

        fs = ctx.holodoppler_config.get("sampling_freq", 37037)
        stride = ctx.holodoppler_config.get("batch_stride", 256)

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)
        
        self.logger.info(f"    - Camera sampling frequency: {fs} Hz, batch stride: {stride}. Effective sampling frequency after accounting for batch stride: {sampling_frequency:.2f} Hz")

        # --- Step 1: Separate mask into branches ---
        if optic_disc_center is not None:
            labeled_vessels, _ = process_masks.get_labeled_vessels(vessel_mask, *optic_disc_center) 
            if len(np.unique(labeled_vessels)) <= 1:
                self.logger.warning("    - No branches detected in the retinal vessel mask. Might be due to faulty optic disc detection. Trying to label vessels without optic disc center.")

        if optic_disc_center is None or len(np.unique(labeled_vessels)) <= 1: 
            labeled_vessels, _ = process_masks.get_labeled_vessels(vessel_mask, mask_optic_disc=False) # If optic disc center is not available or if no branches were detected, label vessels without using optic disc center
        ctx.set("labeled_vessels", labeled_vessels)

        if len(np.unique(labeled_vessels)) <= 1:
            self.logger.warning("    - No branches detected in the retinal vessel mask. Use vessel mask as pre-mask.")
            ctx.set("pre_artery_mask", vessel_mask)
            ctx.set("pre_vein_mask", np.zeros_like(labeled_vessels))
            self._set_research_masks(ctx, None, None)
            ctx.set("branch_signals", np.zeros((0, video.shape[0])))
            return

        # --- Step 2: Compute mean temporal signal for each branch ---
        # ctx.output_manager.output("pulse_analysis", "M0_ff_video", video, "video")
        signals = pulse_analysis.get_filtered_branch_signals(video, labeled_vessels, sampling_frequency)
        ctx.output_manager.output("pulse_analysis", "labeled_vessels", labeled_vessels, "labeled_mask")
        branch_means = np.nanmean(signals, axis=1, keepdims=True)
        branch_means[~np.isfinite(branch_means)] = 0
        centered_signals = signals - branch_means
        centered_signals[~np.isfinite(centered_signals)] = 0
        branch_stds = centered_signals.std(axis=1, keepdims=True)
        degenerate_branches = ~np.isfinite(branch_stds) | (branch_stds <= np.finfo(float).eps)
        if np.any(degenerate_branches):
            self.logger.warning(
                f"    - {np.count_nonzero(degenerate_branches)} branch signal(s) have no usable temporal variation."
            )
            branch_stds[degenerate_branches] = 1
        signals_n = centered_signals / branch_stds
        ctx.set("branch_signals", signals_n)

        # --- Step 3: Correct signals by aligning with median heartbeat ---
        correct_branch_signals = ctx.dopplerview_config.get("Mask").get("CorrectBranchSignals", True)
        if correct_branch_signals:
            beat_period_frames = pulse_analysis.compute_period(signals_n, sampling_frequency)
            if beat_period_frames is None:
                self.logger.warning(
                    "    - Could not estimate a cardiac period from branch signals. Skipping signal correction."
                )
            else:
                beat_period_time = beat_period_frames / sampling_frequency
                bpm = 60 / beat_period_time

                self.logger.info(f"    - Median heartbeat period: {beat_period_time:.2f} seconds ({beat_period_frames} frames) -> {bpm:.2f} bpm.")
                func = partial(pulse_analysis.correct_signal_with_heartbeat, beat_period=beat_period_frames, k=5)
                corrected_signals = np.asarray(ctx.parallel.map(
                    func,
                    signals_n,
                    task_name="branch signal correction",
                ))
                branch_ids = np.unique(labeled_vessels)
                branch_ids = branch_ids[branch_ids > 0]
                for signal_index, branch_id in enumerate(branch_ids):
                    ctx.output_manager.output("pulse_analysis", f"branch_{branch_id}_corrected", (signals_n[signal_index, :], corrected_signals[signal_index, :]), "signal", options={"multiple_signals": True, "legend": ["Original Signal", "Corrected Signal"]})

                signals_n = corrected_signals

        # --- Step 4: Pre-classify arteries and veins ---
        pre_mask_method = ctx.dopplerview_config.get("Mask").get("PreMaskMethod", "clustering")
        if pre_mask_method not in ["clustering", "gradient", "both"]:
            self.logger.info(f"Warning: Invalid PreMaskMethod {pre_mask_method}, defaulting to clustering.")
            pre_mask_method = "clustering"
        methods = ("clustering", "gradient") if pre_mask_method == "both" else (pre_mask_method,)
        masks = {}

        for method in methods:
            if method == "clustering":
                self.logger.info("    - Pre-classifying arteries and veins using clustering on the first harmonic of branch signals in the complex domain")
                artery_mask, vein_mask, labels, z = pulse_analysis.compute_pre_masks_by_clustering(signals_n, labeled_vessels, sampling_frequency)
                ctx.output_manager.save_clusterization("pulse_analysis", "pre_mask_clusterization", labels, z)
            else:
                self.logger.info("    - Pre-classifying arteries and veins using systolic gradient")
                artery_mask, vein_mask = pulse_analysis.compute_pre_masks_by_systolic_gradient(signals_n, labeled_vessels, sampling_frequency)
            masks[method] = (artery_mask, vein_mask)

            if pre_mask_method == "both":
                ctx.output_manager.save_overlay(
                    "pulse_analysis",
                    f"av_overlay_pre_masks_{method}",
                    ctx.require("M0_ff_image"),
                    [artery_mask, vein_mask],
                )

        canonical_method = "clustering" if pre_mask_method == "both" else pre_mask_method
        pre_artery_mask, pre_vein_mask = masks[canonical_method]
        ctx.output_manager.save_overlay("pulse_analysis", "av_overlay_pre_masks", ctx.require("M0_ff_image"), [pre_artery_mask, pre_vein_mask])
        ctx.set("pre_artery_mask", pre_artery_mask)
        ctx.set("pre_vein_mask", pre_vein_mask)
        self._set_research_masks(
            ctx,
            masks.get("clustering") if pre_mask_method == "both" else None,
            masks.get("gradient") if pre_mask_method == "both" else None,
        )

    @staticmethod
    def _set_research_masks(ctx, clustering_masks, gradient_masks):
        for method, method_masks in (
            ("clustering", clustering_masks),
            ("gradient", gradient_masks),
        ):
            artery_mask, vein_mask = method_masks or (None, None)
            ctx.set(f"pre_artery_mask_{method}", artery_mask)
            ctx.set(f"pre_vein_mask_{method}", vein_mask)

class ComputeTemporalCuesStep(BaseStep):
    requires = {
        "M0_ff_video",
        "pre_artery_mask",
        "pre_vein_mask",
        "pre_artery_mask_gradient",
        "choroidal_vessel_mask",
        "LF_M0_ff",
        "HF_M0_ff",
        "band_ratio_ff",
    }
    produces = {"correlation_M0", "diasys_image", "pre_arterial_pulse", "choroidal_pulse", "pre_arterial_pulse_filtered", "choroidal_pulse_filtered", "pre_arterial_pulse_cleaned", "pre_venous_pulse", "pre_venous_pulse_filtered", "M0_ff_image_cleaned", "beat_period", "systole_image", "diastole_image", "systole_index_list", "correlation_LF_M0_ff", "correlation_HF_M0_ff", "correlation_band_ratio_ff", "diasys_LF_M0_ff", "diasys_HF_M0_ff", "diasys_band_ratio_ff", "correlation_M0_clustering", "diasys_image_clustering", "correlation_M0_gradient", "diasys_image_gradient"}
    name = "temporal_cues"

    def _relevant_config(self, ctx):
        return {"sampling_freq": ctx.holodoppler_config.get("sampling_freq", 37037),
                "batch_stride": ctx.holodoppler_config.get("batch_stride", 256)}

    def run(self, ctx):
        video = ctx.require("M0_ff_video")
        pre_artery_mask = ctx.require("pre_artery_mask")
        pre_vein_mask = ctx.require("pre_vein_mask")
        choroidal_vessel_mask = ctx.require("choroidal_vessel_mask")

        # --- Get pulses from masks ---

        if not np.any(pre_artery_mask):
            raise RuntimeError("Cannot compute temporal cues: the preliminary artery mask is empty")
        arterial_pulse = signal_processing.get_pulse_from_mask(video, pre_artery_mask)
        venous_pulse = (
            signal_processing.get_pulse_from_mask(video, pre_vein_mask)
            if np.any(pre_vein_mask)
            else np.zeros(video.shape[0], dtype=float)
        )
        choroidal_pulse = (
            signal_processing.get_pulse_from_mask(video, choroidal_vessel_mask)
            if np.any(choroidal_vessel_mask)
            else np.zeros(video.shape[0], dtype=float)
        )
        ctx.set("pre_arterial_pulse", arterial_pulse)
        ctx.set("pre_venous_pulse", venous_pulse)
        ctx.set("choroidal_pulse", choroidal_pulse)

        # --- Filter pulses to remove high frequency noise ---

        fs = ctx.holodoppler_config.get("sampling_freq", 37037)
        stride = ctx.holodoppler_config.get("batch_stride", 256)

        sampling_frequency = pulse_analysis.get_effective_sampling_frequency(fs, stride)

        arterial_pulse_filtered = signal_processing.get_filtered_pulse(arterial_pulse, sampling_frequency)
        venous_pulse_filtered = signal_processing.get_filtered_pulse(venous_pulse, sampling_frequency)
        choroidal_pulse_filtered = signal_processing.get_filtered_pulse(choroidal_pulse, sampling_frequency)
        ctx.set("pre_arterial_pulse_filtered", arterial_pulse_filtered)
        ctx.set("pre_venous_pulse_filtered", venous_pulse_filtered)
        ctx.set("choroidal_pulse_filtered", choroidal_pulse_filtered)

        # --- Remove bad beats from arterial pulse by comparing to median beat pattern ---

        beat_period_frames = pulse_analysis.compute_period(arterial_pulse_filtered, sampling_frequency)
        if beat_period_frames is None:
            raise RuntimeError(
                "Cannot compute temporal cues: no cardiac period was detected in the arterial pulse"
            )
        ctx.set("beat_period", beat_period_frames)
        beat_period_time = beat_period_frames / sampling_frequency
        bpm = 60 / beat_period_time
        self.logger.info(f"    - Arterial heartbeat period: {beat_period_time:.2f} seconds ({beat_period_frames} frames) -> {bpm:.2f} bpm.")

        if len(arterial_pulse_filtered) // beat_period_frames < 3 :
            self.logger.warning(f"    - Not enough beats detected in arterial pulse for reliable outlier removal (only {len(arterial_pulse_filtered) // beat_period_frames} beat(s) detected). Skipping beat cleaning.")
            arterial_pulse_cleaned = arterial_pulse_filtered
            video_cleaned = video
        else:
            arterial_pulse_cleaned, video_cleaned, beat_signal, median_beat, peaks = pulse_analysis.remove_bad_beats_on_video(arterial_pulse_filtered, video, beat_period_frames, threshold=0.8)
            ctx.output_manager.output("pulse_analysis", f"outlier removal", (arterial_pulse_filtered, beat_signal), "signal", options={"multiple_signals": True, "legend": ["Original Signal", "beat signal"]})
            ctx.output_manager.output("pulse_analysis", f"median beat", median_beat, "signal")
            ctx.output_manager.output("pulse_analysis", f"peaks", (arterial_pulse_filtered, peaks), "signal", options={"scatter": True})
            if (len(arterial_pulse_filtered) - len(arterial_pulse_cleaned)) > (len(arterial_pulse_filtered) / 2):
                self.logger.info(f"    - Outlier frames removal cancelled, as it would remove {len(arterial_pulse_filtered) - len(arterial_pulse_cleaned)}/{len(arterial_pulse_filtered)} frames.")
                arterial_pulse_cleaned, video_cleaned = arterial_pulse_filtered, video  # Revert to original if too many frames would be removed
            else: 
                self.logger.info(f"    - Removed {len(arterial_pulse_filtered) - len(arterial_pulse_cleaned)} frames due to low correlation with median arterial pulse beat pattern")

        ctx.set("pre_arterial_pulse_cleaned", arterial_pulse_cleaned)
        M0_ff_image_cleaned = image_utils.normalize_to_uint8(np.mean(video_cleaned, axis=0))
        ctx.set("M0_ff_image_cleaned", M0_ff_image_cleaned)

        # --- Compute correlation map with filtered pulses ---

        correlation_artery = signal_processing.compute_correlation(video_cleaned, arterial_pulse_cleaned)
        ctx.set("correlation_M0", correlation_artery)
        ctx.output_manager.output("pulse_analysis", f"correlation map RGB", correlation_artery, "image", options={"blue_gray_red": True, "M0_ff_image": M0_ff_image_cleaned})

        # --- Accumulate frames at the systolic and diastolic peaks of the filtered pulses ---
        
        diasys, sysindexes, diasindexes, systole, diastole, sys_index_list = pulse_analysis.compute_diasys_image(video_cleaned, arterial_pulse_cleaned, sampling_frequency)
        ctx.output_manager.output("pulse_analysis", f"diasys image RGB", diasys, "image", options={"blue_gray_red": True, "M0_ff_image": M0_ff_image_cleaned})
        ctx.output_manager.output("pulse_analysis", f"diasys plot", (arterial_pulse_cleaned, sysindexes), "signal", options={"scatter": True})

        ctx.set("systole_index_list", sys_index_list)
        ctx.set("diasys_image", diasys)
        ctx.set("systole_image", systole)
        ctx.set("diastole_image", diastole)

        # --- Repeat for LF_M0_ff, HF_M0_ff, and band_ratio_ff if they exist in the context ---

        LF_M0_ff = ctx.require("LF_M0_ff")
        if LF_M0_ff is not None:
            correlation_LF_M0_ff = signal_processing.compute_correlation(LF_M0_ff, arterial_pulse_filtered)
            M0_Systole_img, M0_Diastole_img = np.nanmean(LF_M0_ff[sysindexes], axis=0), np.nanmean(LF_M0_ff[diasindexes], axis=0)
            diasys_LF_M0_ff = M0_Systole_img - M0_Diastole_img
        else:
            correlation_LF_M0_ff = None
            diasys_LF_M0_ff = None
        ctx.set("correlation_LF_M0_ff", correlation_LF_M0_ff)
        ctx.set("diasys_LF_M0_ff", diasys_LF_M0_ff)

        HF_M0_ff = ctx.require("HF_M0_ff")
        if HF_M0_ff is not None:
            correlation_HF_M0_ff = signal_processing.compute_correlation(HF_M0_ff, arterial_pulse_filtered)
            M0_Systole_img, M0_Diastole_img = np.nanmean(HF_M0_ff[sysindexes], axis=0), np.nanmean(HF_M0_ff[diasindexes], axis=0)
            diasys_HF_M0_ff = M0_Systole_img - M0_Diastole_img
        else:
            correlation_HF_M0_ff = None
            diasys_HF_M0_ff = None
        ctx.set("correlation_HF_M0_ff", correlation_HF_M0_ff)
        ctx.set("diasys_HF_M0_ff", diasys_HF_M0_ff)

        band_ratio_ff = ctx.require("band_ratio_ff")
        if band_ratio_ff is not None:
            correlation_band_ratio_ff = signal_processing.compute_correlation(band_ratio_ff, arterial_pulse_filtered)
            M0_Systole_img, M0_Diastole_img = np.nanmean(band_ratio_ff[sysindexes], axis=0), np.nanmean(band_ratio_ff[diasindexes], axis=0)
            diasys_band_ratio_ff = M0_Systole_img - M0_Diastole_img
        else:
            correlation_band_ratio_ff = None
            diasys_band_ratio_ff = None
        ctx.set("correlation_band_ratio_ff", correlation_band_ratio_ff)
        ctx.set("diasys_band_ratio_ff", diasys_band_ratio_ff)

        pre_mask_method = ctx.dopplerview_config.get("Mask").get("PreMaskMethod", "clustering")
        if pre_mask_method == "both":
            ctx.set("correlation_M0_clustering", correlation_artery)
            ctx.set("diasys_image_clustering", diasys)
            ctx.output_manager.output(
                "pulse_analysis",
                "correlation map RGB clustering",
                correlation_artery,
                "image",
                options={"blue_gray_red": True, "M0_ff_image": M0_ff_image_cleaned},
            )
            ctx.output_manager.output(
                "pulse_analysis",
                "diasys image RGB clustering",
                diasys,
                "image",
                options={"blue_gray_red": True, "M0_ff_image": M0_ff_image_cleaned},
            )
            gradient_correlation, gradient_diasys = self._compute_research_temporal_cues(
                ctx,
                ctx.get("pre_artery_mask_gradient"),
                sampling_frequency,
            )
            ctx.set("correlation_M0_gradient", gradient_correlation)
            ctx.set("diasys_image_gradient", gradient_diasys)
        else:
            ctx.set("correlation_M0_clustering", None)
            ctx.set("diasys_image_clustering", None)
            ctx.set("correlation_M0_gradient", None)
            ctx.set("diasys_image_gradient", None)

    def _compute_research_temporal_cues(self, ctx, artery_mask, sampling_frequency):
        """Compute the additional gradient M0 cues without changing canonical outputs."""
        if artery_mask is None or not np.any(artery_mask):
            self.logger.warning(
                "    - Gradient pre-mask is empty; gradient temporal cues are unavailable."
            )
            return None, None

        video = ctx.require("M0_ff_video")
        arterial_pulse = signal_processing.get_pulse_from_mask(video, artery_mask)
        arterial_pulse_filtered = signal_processing.get_filtered_pulse(
            arterial_pulse,
            sampling_frequency,
        )
        beat_period_frames = pulse_analysis.compute_period(
            arterial_pulse_filtered,
            sampling_frequency,
        )
        if beat_period_frames is None:
            self.logger.warning(
                "    - No cardiac period detected with the gradient pre-mask; "
                "gradient temporal cues are unavailable."
            )
            return None, None

        if len(arterial_pulse_filtered) // beat_period_frames < 3:
            arterial_pulse_cleaned = arterial_pulse_filtered
            video_cleaned = video
        else:
            arterial_pulse_cleaned, video_cleaned, *_ = pulse_analysis.remove_bad_beats_on_video(
                arterial_pulse_filtered,
                video,
                beat_period_frames,
                threshold=0.8,
            )
            if len(arterial_pulse_filtered) - len(arterial_pulse_cleaned) > len(arterial_pulse_filtered) / 2:
                arterial_pulse_cleaned, video_cleaned = arterial_pulse_filtered, video

        cleaned_image = image_utils.normalize_to_uint8(np.mean(video_cleaned, axis=0))
        correlation = signal_processing.compute_correlation(
            video_cleaned,
            arterial_pulse_cleaned,
        )
        diasys, *_ = pulse_analysis.compute_diasys_image(
            video_cleaned,
            arterial_pulse_cleaned,
            sampling_frequency,
        )
        ctx.output_manager.output(
            "pulse_analysis",
            "correlation map RGB gradient",
            correlation,
            "image",
            options={"blue_gray_red": True, "M0_ff_image": cleaned_image},
        )
        ctx.output_manager.output(
            "pulse_analysis",
            "diasys image RGB gradient",
            diasys,
            "image",
            options={"blue_gray_red": True, "M0_ff_image": cleaned_image},
        )
        return correlation, diasys
