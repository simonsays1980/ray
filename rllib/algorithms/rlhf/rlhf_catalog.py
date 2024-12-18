import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import LLMConfig
from ray.rllib.core.models.base import Model
from ray.rllib.utils.annotations import override

class RLHFCatalog(PPOCatalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        self.pi_head_config = LLMConfig(
            model_id=self._model_config_dict["sft_model_id"],
        )
        self.vf_head_config = LLMConfig(
            model_id=self._model_config_dict["rm_model_id"],
            classifier=True,
        )

    @override(PPOCatalog)
    def build_pi_head(self, framework: str) -> Model:
        return self.pi_head_config.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str) -> Model:
        return self.vf_head_config.build(framework=framework)
