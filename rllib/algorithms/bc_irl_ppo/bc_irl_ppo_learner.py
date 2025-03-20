
from ray.rllib.algorithms.marwil.marwil_learner import MARWILLearner
from ray.rllib.utils.annotations import override


class BCIRLPPOLearner(MARWILLearner):

    @override(MARWILLearner)
    def build(self) -> None:
        super().build()
    