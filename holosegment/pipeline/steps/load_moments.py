from holosegment.pipeline.step import BaseStep
from holosegment.input_output.read_moments import Moments

class LoadMomentsStep(BaseStep):
    name = "load_moments"
    produces = {"moments"}

    def _relevant_config(self, ctx):
        # No specific config for this step, but we include input path for fingerprinting
        return { "h5_file": ctx.cache.get("h5_file", "") }

    def run(self, ctx):
        input_path = ctx.require("h5_file")
        reader = Moments(input_path)
        reader.read_moments()
        ctx.cache["moments"] = reader