from typing import Any, Dict

from ray.rllib.algorithms.sac.sac_learner import QF_PREDS, QF_TARGET_PREDS
from ray.rllib.algorithms.sac.sac_rl_module import SACRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict

_, tf, _ = try_import_tf()


class SACTfRLModule(TfRLModule, SACRLModule):
    framework: str = "tf2"

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Dict[str, Any]:
        output = {}

        # Pi encoder forward pass.
        pi_encoder_outs = self.pi_encoder(batch)        

        # Pi head.
        output[SampleBatch.ACTION_DIST_INPUTS] = self.pi(
            pi_encoder_outs[ENCODER_OUT]
        )

        return output

    def _forward_exploration(self, batch: NestedDict) -> Dict[str, Any]:
        self._forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Dict(str, Any):
        output = {}

        # SAC needs also Q function values and action logits for next observations.
        # TODO (simon): Check, if we need to override the Encoder input_sp
        batch_curr = NestedDict({SampleBatch.OBS: batch[SampleBatch.OBS]})
        # TODO (sven): If we deprecate 'SampleBatch.NEXT_OBS` we need to change 
        # # this. 
        batch_next = NestedDict({SampleBatch.OBS: batch[SampleBatch.NEXT_OBS]})
        
        # Encoder forward passes.
        pi_encoder_outs = self.pi_encoder(batch_curr)
        batch_curr.update({SampleBatch.ACTIONS: batch[SampleBatch.ACTIONS]})
        qf_encoder_outs = self.qf_encoder(batch_curr)
        qf_target_encoder_outs = self.qf_target_encoder(batch_curr)
        # Also encode the next observations (and next actions for the Q net).
        pi_encoder_next_outs = self.pi_encoder(batch_next)
        # TODO (simon): Make sure these are available.
        batch_next.update({SampleBatch.ACTIONS: batch["next_actions"]})
        qf_encoder_nect_outs = self.qf_encoder(batch_next)

        # Q heads.
        qf_out = self.qf(qf_encoder_outs[ENCODER_OUT])
        qf_target_out = self.qf_target(qf_target_encoder_outs[ENCODER_OUT])
        # Also get the Q-value for the next observations.
        qf_next_out = self.qf(pi_encoder_next_outs[ENCODER_OUT])
        # Squeeze out last dimension (Q function node).
        output[QF_PREDS] = tf.squeeze(qf_out, axis=-1)
        output[QF_TARGET_PREDS] = tf.squeeze(qf_target_out, axis=-1)
        output["qf_preds_next"] = tf.squeeze(qf_next_out, axis=-1)

        # Policy head.
        action_logits = self.pi(pi_encoder_outs[ENCODER_OUT])
        # Also get the action logits for the next observations.
        action_logits_next = self.pi(pi_encoder_next_outs[ENCODER_OUT])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        output["action_dist_inputs_next"] = action_logits_next

        # Return the network outputs.
        return output
