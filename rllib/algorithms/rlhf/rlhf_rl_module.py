import abc
from typing import Any, Dict


from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class RLHFRLModule(RLModule, ValueFunctionAPI, abc.ABC):
    @override(RLModule)
    def setup(self):
        
        self.pad_token_id = self.model_config["pad_token_id"]
        self.generation_config = self.model_config["generation_config"]

        self.temperature = self.model_config["temperature"]

        if self.catalog is None and hasattr(self, "_catalog_ctor_error"):
            raise self._catalog_ctor_error

        # TODO (simon): This is actually not true for the LLM, but we want to
        # avoid RLlib to plug in its stateful connectors. Find another way to
        # to do it.
        # is_stateful = False

        # Build models from catalog.
        self.pi = self.catalog.build_pi_head(framework=self.framework)
        self.vf = self.catalog.build_vf_head(framework=self.framework)

        self.ref_pi = self.catalog.build_pi_head(framework=self.framework)
        self.rm = self.catalog.build_vf_head(framework=self.framework)

    # TODO (simon): In the long run create another API for reward models.
    @abc.abstractmethod
    def compute_rewards(self, batch: Dict[str, Any]) -> TensorType:
        return NotImplemented
