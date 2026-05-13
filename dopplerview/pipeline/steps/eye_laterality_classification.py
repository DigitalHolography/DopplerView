from dopplerview.pipeline.step import BaseStep

import numpy as np

class EyeLateralityClassificationStep(BaseStep):
    requires = {"M0_ff_image_cleaned"}
    produces = {"eye_laterality", "eye_laterality_confidence"}
    name = "eye_laterality_classification"

    def _relevant_config(self, ctx):
        return {}
    
    def classify(self, ctx):
        model = ctx.get_current_model_for_task(self.name)

        input = model.prepare_input(ctx)
        output = np.squeeze(model.predict(input))
        pred = np.argmax(output)
        confidence = output[pred]

        laterality = "left" if pred == 0 else "right"

        self.logger.info(f"    - Eye laterality classified as: {laterality} (confidence: {confidence:.2f})")

        return pred, confidence
    
    def run(self, ctx):
        laterality, confidence = self.classify(ctx)
        ctx.set("eye_laterality", laterality)
        ctx.set("eye_laterality_confidence", confidence)