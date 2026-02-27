from holosegment.steps.step import BaseStep
from holosegment.input_output.read_moments import Moments

class LoadMomentsStep(BaseStep):
    name = "load_moments"
    produces = ["moments"]

    def run(self, ctx):
        input_path = ctx.require("input_path")
        reader = Moments(input_path)
        reader.read_moments()
        ctx.cache["moments"] = reader