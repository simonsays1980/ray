from typing import Optional

from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.ppo import PPOConfig, PPO

class RLHFConfig(PPOConfig):

    def __init__(self, algo_class=None):

        super().__init__(algo_class=algo_class or PPO)
        # The SFT model ID on huggingface.
        self.sft_model_id: Optional[str] = None

    def training(
        self,
        *,
        sft_model_id: Optional[str] = NotProvided,
        **kwargs,
    ) -> "RLHFConfig":

        super().training(**kwargs)

        if sft_model_id is not NotProvided:
            self.sft_model_id = sft_model_id

        return self
        
        


