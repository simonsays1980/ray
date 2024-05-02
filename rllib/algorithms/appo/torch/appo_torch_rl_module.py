from typing import List

from ray.rllib.algorithms.appo.appo import (
    OLD_ACTION_DIST_LOGITS_KEY,
)
from ray.rllib.algorithms.appo.appo_rl_module import APPORLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR
from ray.rllib.core.models.tf.encoder import ENCODER_OUT
from ray.rllib.core.rl_module.rl_module_with_target_networks_interface import (
    RLModuleWithTargetNetworksInterface,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict


class APPOTorchRLModule(
    PPOTorchRLModule, RLModuleWithTargetNetworksInterface, APPORLModule
):
    @override(RLModuleWithTargetNetworksInterface)
    def get_target_network_pairs(self):
        return [(self.old_pi, self.pi), (self.old_encoder, self.encoder)]

    @override(PPOTorchRLModule)
    def output_specs_train(self) -> List[str]:
        return [
            Columns.ACTION_DIST_INPUTS,
            OLD_ACTION_DIST_LOGITS_KEY,
            Columns.VF_PREDS,
        ]

    @override(PPOTorchRLModule)
    def _forward_train(self, batch: NestedDict):
        if self.inference_only:
            raise RuntimeError(
                "Trying to train a module that is not a learner module. Set the "
                "flag `inference_only=False` when building the module."
            )

        outs = super()._forward_train(batch)
        old_pi_inputs_encoded = self.old_encoder(batch)[ENCODER_OUT][ACTOR]
        old_action_dist_logits = self.old_pi(old_pi_inputs_encoded)
        outs[OLD_ACTION_DIST_LOGITS_KEY] = old_action_dist_logits
        return outs

    @override(PPOTorchRLModule)
    def _set_inference_only_state_dict_keys(self) -> None:
        # Get the model_parameters from the `PPOTorchRLModule`.
        super()._set_inference_only_state_dict_keys()
        # Get the model_parameters.
        state_dict = self.state_dict()
        # Note, these keys are only known to the learner module. Furthermore,
        # we want this to be run once during setup and not for each worker.
        self._inference_only_state_dict_keys["unexpected_keys"].extend(
            [name for name in state_dict if "old" in name]
        )
