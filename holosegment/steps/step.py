from typing import List

class BaseStep:
    """
    Base class for pipeline steps.

    Each step must define:
        - name
        - requires (list of data keys)
        - produces (list of data keys)
    """

    name: str = None
    requires: List[str] = []
    produces: List[str] = []

    def run(self, ctx):
        raise NotImplementedError
    
class NestedStep(BaseStep):
    substeps: List[BaseStep] = []

    def run(self, ctx):
        for step in self.substeps:
            step.run(ctx)