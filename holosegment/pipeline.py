"""
Pipeline for Holosegment.
"""

from holosegment.io.read_moments import Moments
from holosegment.preprocessing.preprocessing import Preprocessor

class Pipeline:
    def __init__(self, config, cache, model_registry):
        self.config = config
        self.cache = cache
        self.models = model_registry

    def run(self, input_path):
        # Step 1: Load moments data
        moments = self.load_moments(input_path)
        self.cache.M0 = moments.M0  # Cache M0 data for use in segmentation

        # Step 2: Preprocess frames (normalization, registration and flatfield correction)
        self.preprocess(moments)

        # Step 3: Perform binary vessel segmentation
        vessel_mask = self.segment_vessels(preprocessed_moments)

        # Step 4: Perform pulse analysis to compute correlation map and diasys map 


        pulse = self.pulse_analysis(preprocessed_moments, vessel_mask)
        av_mask = self.av_segmentation(preprocessed_moments, pulse)
        return av_mask
    
    def load_moments(self, input_path):
        reader = Moments(input_path)
        reader.read_moments()  # Load data into reader.M0, reader.M1, reader.M2, reader.SH
        return reader
    
    def preprocess(self, moments):
        preprocessor = Preprocessor(self.config)
        preprocessor.preprocess(moments)
        self.cache.M0_ff_video = preprocessor.M0_ff_video  # Cache flatfield-corrected M0 video
        self.cache.M0_ff_image = preprocessor.M0_ff_image  # Cache flatfield

    def segment_vessels(self, moments):
        # Implement vessel segmentation using the appropriate model from self.models
        pass

    def pulse_analysis(self, moments, vessel_mask):
        # Implement pulse analysis to compute correlation map and diasys map
        pass

    def av_segmentation(self, moments, pulse):
        # Implement artery vein segmentation using the appropriate model from self.models
        pass
