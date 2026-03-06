from holosegment.pipeline.step import BaseStep
from holosegment.input_output.read_moments import Moments

class LoadMomentsStep(BaseStep):
    name = "load_moments"
    produces = {"moments"}

    def _relevant_config(self, ctx):
        # No specific config for this step, but we include input path for fingerprinting
        return { "input_path": ctx.cache.get("input_path", "") }

    def run(self, ctx):
        input_path = ctx.require("input_path")
        reader = Moments(input_path)
        reader.read_moments()
        ctx.cache["moments"] = reader