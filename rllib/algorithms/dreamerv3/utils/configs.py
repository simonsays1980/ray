from dataclasses import dataclass
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import _framework_implemented, _MLPConfig
from ray.rllib.utils.annotations import ExperimentalAPI, override


@ExperimentalAPI
@dataclass
class DynamicsPredictorConfig(_MLPConfig):

    num_categoricals: int = 32
    num_classes_per_categorical: int = 32

    @override(_MLPConfig)
    def _validate(self, framework: str = "torch"):
        """Makes sure categoricals and classes are integers."""

        assert isinstance(self.num_categoricals, int)
        assert isinstance(self.num_classes_per_categorical, int)

        return super()._validate(framework)

    @_framework_implemented()
    def build(self, framework: str = "torch") -> "Model":
        self._validate(framework=framework)

        if framework == "torch":
            from ray.rllib.algorithms.dreamerv3.torch.models.torch_dynamics_predictor import (
                TorchDynamicsPredictor,
            )

            return TorchDynamicsPredictor(config=self)
        else:
            # TODO (simon, sven): Shall we add a TF implementation here?
            raise ValueError(f"Framework {framework} not supported!")
