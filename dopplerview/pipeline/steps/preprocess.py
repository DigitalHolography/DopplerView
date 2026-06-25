from dopplerview.pipeline.step import BaseStep

from dopplerview.preprocessing import normalization
from dopplerview.utils import image_utils

import numpy as np

class PreprocessStep(BaseStep):
    requires = {"moment0", "moment1", "moment2", "HF_M0", "LF_M0"}
    produces = {"M0_ff_video", "M0_ff_image", "M1_ff_video", "M1_ff_image", "M2_ff_video", "M2_ff_image", "HF_M0_ff", "HF_M0_ff_image", "LF_M0_ff", "LF_M0_ff_image", "band_ratio_ff", "band_ratio_ff_image"}
    name = "preprocess"

    def _relevant_config(self, ctx):
        return {
            "NumberOfWorkers": ctx.dopplerview_config.get("NumberOfWorkers", 0.5),
            "FlatFieldCorrection": {
                "GWRatio": ctx.dopplerview_config.get("FlatFieldCorrection", {}).get("GWRatio", 0.07),
            }
        }

    def normalize(self, gaussian_std, M0, M1, M2, HF_M0, LF_M0, n_jobs=-1):
        # Implement normalization logic based on self.dopplerview_config
        # self.logger.info(self.dopplerview_config)

        numx = M0.shape[2]
        M0_ff_video = normalization.flat_field_correction_3d(M0, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        M1_ff_video = normalization.flat_field_correction_3d(M1, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        M2_ff_video = normalization.flat_field_correction_3d(M2, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        HF_M0_ff = normalization.flat_field_correction_3d(HF_M0, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        LF_M0_ff = normalization.flat_field_correction_3d(LF_M0, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        return M0_ff_video, M1_ff_video, M2_ff_video, HF_M0_ff, LF_M0_ff
    
    def resize(self):
        # Implement resizing logic based on self.dopplerview_config
        return
    
    def remove_outliers(self):
        # Implement outlier removal logic based on self.dopplerview_config
        return
    
    def interpolate(self):
        # Implement interpolation logic based on self.dopplerview_config
        return

    def run(self, ctx):

        moment0 = ctx.require("moment0")
        moment1 = ctx.require("moment1")
        moment2 = ctx.require("moment2")
        HF_M0 = ctx.require("HF_M0")
        LF_M0 = ctx.require("LF_M0")

        # Step 1: Normalize 
        self.logger.info("    - Applying flat field correction to the moments")
        gaussian_std = ctx.dopplerview_config.get("FlatFieldCorrection", {}).get("GWRatio", 0.07)
        n_jobs = ctx.dopplerview_config.get("NumberOfWorkers", 0.5)
        M0_ff_video, M1_ff_video, M2_ff_video, HF_M0_ff, LF_M0_ff = self.normalize(gaussian_std, moment0, moment1, moment2, HF_M0, LF_M0, n_jobs=n_jobs)

        band_ratio_ff = np.divide(HF_M0_ff, LF_M0_ff, out=np.zeros_like(HF_M0_ff), where=LF_M0_ff!=0)
        # # Step 2: Resize
        # self.resize()

        # # Step 3: Interpolate
        # self.interpolate()

        # # Step 4: Remove outliers 
        # self.remove_outliers()
        ctx.set("M0_ff_video", M0_ff_video)
        ctx.set("M1_ff_video", M1_ff_video)
        ctx.set("M2_ff_video", M2_ff_video)
        ctx.set("M0_ff_image", image_utils.normalize_to_uint8(np.mean(M0_ff_video, axis=0)) if M0_ff_video is not None else None)
        ctx.set("M1_ff_image", image_utils.normalize_to_uint8(np.mean(M1_ff_video, axis=0)) if M1_ff_video is not None else None)
        ctx.set("M2_ff_image", image_utils.normalize_to_uint8(np.mean(M2_ff_video, axis=0)) if M2_ff_video is not None else None)
        ctx.set("HF_M0_ff", HF_M0_ff)
        ctx.set("LF_M0_ff", LF_M0_ff)
        ctx.set("band_ratio_ff", band_ratio_ff)
        ctx.set("HF_M0_ff_image", image_utils.normalize_to_uint8(np.mean(HF_M0_ff, axis=0)) if HF_M0_ff is not None else None)
        ctx.set("LF_M0_ff_image", image_utils.normalize_to_uint8(np.mean(LF_M0_ff, axis=0)) if LF_M0_ff is not None else None)
        ctx.set("band_ratio_ff_image", image_utils.normalize_to_uint8(np.mean(band_ratio_ff, axis=0)) if band_ratio_ff is not None else None)