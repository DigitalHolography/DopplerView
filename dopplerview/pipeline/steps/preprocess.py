from dopplerview.pipeline.step import BaseStep

from dopplerview.preprocessing.registration import register_video
from dopplerview.preprocessing import normalization, resize
from dopplerview.utils import image_utils
from dopplerview.utils.parallelization_utils import run_in_parallel

from functools import partial
import numpy as np

class PreprocessStep(BaseStep):
    requires = {"moment0", "moment1", "moment2"}
    produces = {"M0_ff_video", "M0_ff_image", "M1_ff_video", "M1_ff_image", "M2_ff_video", "M2_ff_image"}
    name = "preprocess"

    def _relevant_config(self, ctx):
        return {
            # "Preprocess": {
            #     "Register": ctx.dopplerview_config["Preprocess"]["Register"],
            #     "Crop": ctx.dopplerview_config["Preprocess"]["Crop"]
            # },
            "NumberOfWorkers": ctx.dopplerview_config["NumberOfWorkers"],
            "FlatFieldCorrection": {
                "GWRatio": ctx.dopplerview_config["FlatFieldCorrection"]["GWRatio"]
            }
        }

    def normalize(self, gaussian_std, M0, M1, M2, n_jobs=-1):
        # Implement normalization logic based on self.dopplerview_config
        # self.logger.info(self.dopplerview_config)

        numx = M0.shape[2]
        M0_ff_video = normalization.flat_field_correction_3d(M0, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        M1_ff_video = normalization.flat_field_correction_3d(M1, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        M2_ff_video = normalization.flat_field_correction_3d(M2, gaussian_std * numx, parallel=True, n_jobs=n_jobs) # TODO: add parameter for parallelization 

        return M0_ff_video, M1_ff_video, M2_ff_video
    
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

        # Step 1: Normalize 
        self.logger.info("    - Applying flat field correction to the moments")
        gaussian_std = ctx.dopplerview_config['FlatFieldCorrection']['GWRatio']
        n_jobs = ctx.dopplerview_config["NumberOfWorkers"]
        M0_ff_video, M1_ff_video, M2_ff_video = self.normalize(gaussian_std, moment0, moment1, moment2, n_jobs=n_jobs)

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