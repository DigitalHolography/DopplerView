from dopplerview.pipeline.step import BaseStep
import dopplerview.segmentation.process_masks as process_masks
import dopplerview.segmentation.pulse_analysis as pa
import dopplerview.segmentation.clustering as clustering
import dopplerview.segmentation.embedding as embedding
from functools import partial
from skimage.morphology import dilation, opening, disk
from skimage.measure import label
import numpy as np

class RetinalAVSegmentationStep(BaseStep):
    requires = {"M0_ff_video", "M0_ff_image_cleaned", "correlation_M0", "diasys_image", "optic_disc_center"}
    produces = {"retinal_artery_mask", "retinal_vein_mask", "retinal_artery_mask_clean", "retinal_vein_mask_clean"}
    name = "retinal_artery_vein_segmentation"

    def _relevant_config(self, ctx):
        params = ctx.dopplerview_config["Mask"]
        return { "AVSegmentationMethod": params.get("AVSegmentationMethod", "AI"),
                    "av_segmentation_model_name": ctx.get_current_model_name_for_task(self.name),
                    "DiaphragmRadius": params.get("DiaphragmRadius", 0.45),
                    "CenterRadius": params.get("CenterRadius", 0.1)
        }

    def deep_segmentation(self, ctx):
        model = ctx.get_current_model_for_task(self.name)

        input = model.prepare_input(ctx)

        mask = model.predict(input)
        mask = np.squeeze(mask)  # Remove channel dimension if present

        if model.spec.output_activation == "argmax":
            return np.where((mask==1) | (mask==3), 1, 0), np.where((mask==2) | (mask==3), 1, 0)
        
        return mask[0], mask[1]

    def handmade_segmentation(self, ctx):
        raise NotImplementedError("Handmade artery vein segmentation not implemented yet.")
    
    def clean_mask(self, raw_mask, ctx):
        optic_disc_center = ctx.require("optic_disc_center")
        if optic_disc_center is None:
            self.logger.warning("    - Optic disc center not available. Skipping mask cleaning.")
            return raw_mask
        
        width, height = raw_mask.shape
        optic_disc_center = (optic_disc_center[0] / width, optic_disc_center[1] / height)

        params = ctx.dopplerview_config["Mask"]

        clean_mask = process_masks.clean_vessel_mask(
            raw_mask,
            image_shape=raw_mask.shape,
            optic_disc_center=optic_disc_center,
            diaphragm_radius=params.get("DiaphragmRadius", 0.45),
            connect_radius=params.get("CenterRadius", 0.1),
            tolerance=5
        )

        return clean_mask

    def run(self, ctx):
        if ctx.dopplerview_config.get("AVSegmentationMethod", "AI") == "AI":
            self.logger.info("    - Use deep segmentation model for artery vein segmentation.")
            artery_mask, vein_mask = self.deep_segmentation(ctx)
        else:
            self.logger.info("    - Use hand-made heuristics for artery vein segmentation.")
            artery_mask, vein_mask = self.handmade_segmentation(ctx)

        ctx.set("retinal_artery_mask", artery_mask)
        ctx.set("retinal_vein_mask", vein_mask)
        ctx.output_manager.save_overlay(self.name, "av_overlay", ctx.require("M0_ff_image_cleaned"), [artery_mask, vein_mask])

        artery_mask_clean = self.clean_mask(artery_mask, ctx)
        self.logger.info(f"    - Artery mask: {artery_mask.sum()} pixels, Cleaned artery mask: {artery_mask_clean.sum()} pixels")
        vein_mask_clean = self.clean_mask(vein_mask, ctx)
        self.logger.info(f"    - Vein mask: {vein_mask.sum()} pixels, Cleaned vein mask: {vein_mask_clean.sum()} pixels")

        ctx.set("retinal_artery_mask_clean", artery_mask_clean)
        ctx.set("retinal_vein_mask_clean", vein_mask_clean)
        ctx.output_manager.save_overlay(self.name, "av_overlay_cleaned", ctx.require("M0_ff_image_cleaned"), [artery_mask_clean, vein_mask_clean])

class ChoroidalAVSegmentationStep(BaseStep):
    requires = {"M0_ff_video", "M0_ff_image_cleaned", "HF_M0_FF", "retinal_artery_mask", "retinal_vein_mask", "correlation_M0", "correlation_HF_M0_ff", "optic_disc_center", "choroidal_vessel_mask"}
    produces = {"choroidal_artery_mask", "choroidal_vein_mask", "choroidal_aliased_artery_mask", "choroidal_artery_mask_clean", "choroidal_vein_mask_clean", "choroidal_aliased_artery_mask_clean"}
    name = "choroidal_artery_vein_segmentation"

    def _relevant_config(self, ctx):
        return {
            "sampling_freq": ctx.holodoppler_config.get("sampling_freq", 37037),
            "ChoroidalSegmentationMethod": ctx.dopplerview_config.get("ChoroidalSegmentationMethod", "clustering"),
        }
    
    def clean_vessel_mask(self, raw_mask, ctx):
        params = ctx.dopplerview_config["Mask"]

        diaphragm_radius = params.get("DiaphragmRadius", 0.45)
        h, w = raw_mask.shape
        mask_diaphragm = process_masks.disk_mask(h, w, diaphragm_radius)

        clean_mask = raw_mask & mask_diaphragm

        # If optic disc center is available, remove vessels in the center region
        optic_disc_center = ctx.require("optic_disc_center")
        if optic_disc_center is not None:
            mask_center = ctx.require("optic_disc_mask")
            clean_mask = clean_mask & ~mask_center

            # If optic disc center is available, retinal masks should be valid, and can be removed
            retinal_artery_mask = ctx.require("retinal_artery_mask")
            retinal_vein_mask = ctx.require("retinal_vein_mask")
            retinal_vessel_mask = retinal_artery_mask | retinal_vein_mask
            dilated_retinal_vessel_mask = dilation(retinal_vessel_mask)
            clean_mask = clean_mask & ~dilated_retinal_vessel_mask

        return clean_mask
    
    def get_retinal_artery_mask_treshold(self, ctx):
        # If the retinal artery mask is valid, return it
        if ctx.require("retinal_artery_mask") is not None:
            if ctx.require("retinal_artery_mask").sum() > 0 and ctx.require("optic_disc_center") is not None:
                return ctx.require("retinal_artery_mask")

        corr_HF = ctx.require("correlation_HF_M0_ff")
        corr_M0 = ctx.require("correlation_M0")

        retinal_artery_mask = (corr_HF > 0.3) | (corr_M0 > 0.35)

        return retinal_artery_mask
    
    def get_retinal_vein_mask_treshold(self, ctx):
        # If the retinal vein mask is valid, return it
        if ctx.require("retinal_vein_mask") is not None:
            if ctx.require("retinal_vein_mask").sum() > 0 and ctx.require("optic_disc_center") is not None:
                return ctx.require("retinal_vein_mask")

        return np.zeros_like(ctx.require("M0_ff_image_cleaned"), dtype=np.uint8)

    def get_masks_treshold(self, ctx):
        # Get artery and vein masks based on the correlation of the M0 and its high frequencies with the retinal aterial signal.
        # The choroidal vessels can be identified as three types:
        # - arteries with positive correlation with the retinal arteries
        # - veins with negative correlation with the retinal arteries, less present in the high frequency signal
        # - aliased arteries with almost perfect negative correlation with the retinal arteries, dominant in the high frequency signal

        corr_M0 = ctx.require("correlation_M0")
        corr_HF = ctx.require("correlation_HF_M0_ff")

        retinal_artery_mask = self.get_retinal_artery_mask(ctx)
        retinal_vein_mask = self.get_retinal_vein_mask(ctx)
        retinal_vessel_mask = dilation(retinal_artery_mask | retinal_vein_mask)

        self.logger.info("    - Identifying choroidal aliased arteries with -0.2 correlation threshold in high frequency signal and -0.25 for M0 correlation.")
        pre_aliased_arteries = corr_HF < -0.2   # Identify aliased arteries based on high frequency correlation
        connected_arteries = process_masks.connect_components(
            pre_aliased_arteries,
            max_distance=5,
        )  # Connect disconnected aliased arteries that are close to each other
        large_arteries = opening(connected_arteries, disk(2))
        anti_correlated_vessels = corr_M0 < -0.25
        aliased_arteries = process_masks.keep_connected_components(anti_correlated_vessels, large_arteries) # Keep only the vessels that are connected to the aliased arteries
        choroidal_aliased_artery_mask = aliased_arteries & ~retinal_vessel_mask  # Remove retinal vessels

        self.logger.info("    - Identifying choroidal veins with -0.15 correlation threshold in M0 signal.")
        pre_veins = corr_M0 < -0.15
        veins = process_masks.keep_connected_components(pre_veins, aliased_arteries, negative=True)   # Remove aliased arteries from the vein mask
        veins = process_masks.remove_small_vessels(label(veins), min_size=10) > 0
        choroidal_vein_mask = veins & ~retinal_vessel_mask  # Remove retinal vessels

        self.logger.info("    - Identifying choroidal arteries with 0.12 correlation threshold in M0 signal.")
        arteries = corr_M0 > 0.12
        arteries = process_masks.remove_small_vessels(label(arteries), min_size=10) > 0
        choroidal_artery_mask = arteries & ~retinal_vessel_mask  # Remove retinal vessels

        return choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask

    def get_masks_clustering(self, ctx):
        # Get artery and vein masks based on clustering of the branch signals.
        # The choroidal vessels can be identified as three types:
        # - arteries with positive correlation with the retinal arteries
        # - veins with negative correlation with the retinal arteries, less present in the high frequency signal
        # - aliased arteries with almost perfect negative correlation with the retinal arteries, dominant in the high frequency signal

        # --- Step 1: Separate mask into branches and extract signals ---
        choroid_vessels = ctx.require("choroidal_vessel_mask")
        optic_disc_center = ctx.get("optic_disc_center")
        M0_ff_video = ctx.require("M0_ff_video")
        sampling_freq = ctx.holodoppler_config.get("sampling_freq", 37037)
        beat_period = ctx.get("beat_period")
        retinal_artery_mask = ctx.require("retinal_artery_mask")
        HF_M0_ff = ctx.require("HF_M0_FF")
        LF_M0_ff = ctx.require("LF_M0_FF")

        if optic_disc_center is not None:
            labeled_vessels_choroid, _ = process_masks.get_labeled_vessels(choroid_vessels, *optic_disc_center) 
            if len(np.unique(labeled_vessels_choroid)) <= 1:
                self.logger.warning("    - No branches detected in the retinal vessel mask. Might be due to faulty optic disc detection. Trying to label vessels without optic disc center.")

        if optic_disc_center is None or len(np.unique(labeled_vessels_choroid)) <= 1: 
            labeled_vessels_choroid, _ = process_masks.get_labeled_vessels(choroid_vessels, mask_optic_disc=False) # If optic disc center is not available or if no branches were detected, label vessels without using optic disc center
        ctx.set("labeled_vessels_choroid", labeled_vessels_choroid)

        signals_choroid = pa.get_filtered_branch_signals(M0_ff_video, labeled_vessels_choroid, sampling_freq)

        # --- Step 2: Cluster signals using complex Fourier embedding ---
        complex_fourier_embedding = partial(embedding.complex_fourier_embedding, sampling_freq=sampling_freq, n_harmonics=3)
        component_names = ["real", "imag"]
        kmeans_2 = partial(clustering.kmeans_clustering, n_clusters=2)

        result_complex_fourier_choroid = clustering.run_clustering_pipeline(
            signals_choroid,
            labeled_vessels_choroid,
            sampling_freq,
            component_names=component_names,
            embedding_func=complex_fourier_embedding,
            clustering_func=kmeans_2,
            video=M0_ff_video,
            correct_signals=False,
            beat_period=beat_period
        )

        ctx.output_manager.save_overlay(self.name, "clusterization_choroid_fourier_embedding", ctx.require("M0_ff_image_cleaned"), [result_complex_fourier_choroid.artery_mask, result_complex_fourier_choroid.vein_mask], colors=[(255, 0, 0), (0, 0, 255)])

        # --- Step 3: Separate aliased arteries from veins using correlation with the retinal arterial signal in high and low frequencies ---
        pseudo_choroid_vein_mask = result_complex_fourier_choroid.vein_mask
        pseudo_labeled_veins_choroid, _ = process_masks.get_labeled_vessels(pseudo_choroid_vein_mask, mask_optic_disc=True)
        choroid_artery_mask = result_complex_fourier_choroid.artery_mask

        corr_stacks_2bands_pixel = pa.correlation_stack_per_pixel(retinal_artery_mask, [HF_M0_ff, LF_M0_ff], pseudo_labeled_veins_choroid, include_std=False)
        agglo_2 = partial(clustering.agglomerative_cluster, n_clusters=2)
        component_names = ["HF_M0 correlation", "LF_M0 correlation"]

        result_correlation_2bands_choroid = clustering.run_clustering_pipeline(
            corr_stacks_2bands_pixel,
            pseudo_labeled_veins_choroid,
            sampling_freq,
            embedding_func=None,
            clustering_func=agglo_2,
            video=M0_ff_video,
            correct_signals=False,
            beat_period=beat_period,
            assign_to_av=False
        )

        choroid_aliased_artery_mask, choroid_vein_mask, mask_labels = pa.assign_corr_stack_to_av(result_correlation_2bands_choroid.X, result_correlation_2bands_choroid.cluster_labels, pseudo_labeled_veins_choroid, negative=True)
        result_correlation_2bands_choroid.artery_mask, result_correlation_2bands_choroid.vein_mask, result_correlation_2bands_choroid.mask_labels = choroid_aliased_artery_mask, choroid_vein_mask, mask_labels
        ctx.output_manager.save_overlay(self.name, "clusterization_choroid_fourier_embedding", ctx.require("M0_ff_image_cleaned"), [choroid_aliased_artery_mask, choroid_vein_mask], colors=[(255, 0, 0), (0, 0, 255)])

        return choroid_artery_mask, choroid_vein_mask, choroid_aliased_artery_mask

    def run(self, ctx):
        # Implement choroidal artery vein segmentation logic based on ctx and self.dopplerview_config
        # For now, we will just set empty masks as placeholders
        segmentation_method = ctx.dopplerview_config.get("ChoroidalSegmentationMethod", "clustering")
        if segmentation_method == "clustering":
            self.logger.info("    - Segment choroidal arteries and veins using clustering of branch signals.")
            choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask = self.get_masks_clustering(ctx)
        elif segmentation_method == "threshold":
            self.logger.info("    - Use retinal arterial correlation thresholding on different frequency bands for choroidal artery vein segmentation.")
            choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask = self.get_masks_treshold(ctx)
        else:
            raise ValueError(f"Unknown ChoroidalSegmentationMethod: {segmentation_method}")

        ctx.set("choroidal_artery_mask", choroidal_artery_mask)
        ctx.set("choroidal_vein_mask", choroidal_vein_mask)
        ctx.set("choroidal_aliased_artery_mask", choroidal_aliased_artery_mask)

        ctx.output_manager.save_overlay(self.name, "overlay", ctx.require("M0_ff_image_cleaned"), [choroidal_artery_mask, choroidal_vein_mask, choroidal_aliased_artery_mask], colors=[(0, 0, 255), (255, 0, 0), (0, 255, 0)])

        choroidal_artery_mask_clean = self.clean_vessel_mask(choroidal_artery_mask, ctx)
        choroidal_vein_mask_clean = self.clean_vessel_mask(choroidal_vein_mask, ctx)
        choroidal_aliased_artery_mask_clean = self.clean_vessel_mask(choroidal_aliased_artery_mask, ctx)

        ctx.set("choroidal_artery_mask_clean", choroidal_artery_mask_clean)
        ctx.set("choroidal_vein_mask_clean", choroidal_vein_mask_clean)
        ctx.set("choroidal_aliased_artery_mask_clean", choroidal_aliased_artery_mask_clean)
        
        ctx.output_manager.save_overlay(self.name, "overlay_clean", ctx.require("M0_ff_image_cleaned"), [choroidal_artery_mask_clean, choroidal_vein_mask_clean, choroidal_aliased_artery_mask_clean], colors=[(0, 0, 255), (255, 0, 0), (0, 255, 0)])
