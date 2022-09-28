from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch.nn import nn

from ray.rllib2.algorithms.callbacks import (
    CallbacksConfig,
    RLlibCallbacks
)
from ray.rllib2.models.configs import ModelConfig
from ray.rllib2.models.torch.encoder import ModelWithEncoder

ModuleType = Union[nn.Module, "tf.Module"]
@dataclass
class RNDCallbacksConfig(CallbacksConfig):
    is_trained: bool = True    
    # @kourosh: We could also use a class inheritance from ModelConfig
    # instead. But imo this approach here is cleaner, as it 
    # is a Callback.
    distill_net_config: ModelConfig = None
    # Embedding dimension for the distillation.
    embedding_dim: int = 128
    # Optimizer config.
    optimizer_config: 
    # Learning rate for the distillation optimizer.
    lr: float = 1e-5
    # If a nonepisodic value head should be used.
    non_episodic_returns: bool = False
    gamma: float = 0.99
    lambda_: float = 0.95
    vf_loss_coeff: float = 0.4
    adv_int_coeff: float = 0.04
    adv_ext_coeff: float = 0.02
    # If intrinsic reward should be normalized. 
    normalize: bool = True
    # Exploration to use for actions.
    sub_exploration: CallbacksConfig = None
    

class RNDExplorationCallbacks(RLlibCallbacks):
    
    def __init__(self, config: CallbackConfig):
        # Initialize super.
        super().__init__(config=config)
        self.config = config
        # Get the config for the distillation networks.
        self.distill_net_config = config.get("distill_net_config").copy()        
        self.embedding_dim = config.get("embedding_dim")        
        
        # @kourosh: Here it would be good to make the encoder of Pi, QF, 
        # or VF net available. So parameter sharing could be used in 
        # the `distill_net` (not the distill_Target_net).
        self.distill_net = ModelWithEncoder(self.distill_net_config)
        self.distill_target_net = ModelWithEncoder(self.distill_target_net)
        
        
        self.non_episodic_returns = self.config.non_episodic_returns
        if self.non_episodic_returns:
            # @kourosh: Get access to the encoder of the Pi, VF, or QF. 
            # This is necessary for the RND algorithm to work properly.
            # Furthermore, this should also be optimized by the RLModule's 
            # optimizer corresponding to the Pi, QF, or VF function.                        
            self.non_episodic_vf = model_catalog.make_vf(
                obs_space=self.config.obs_space,
                action_space=self.config.action_space,
                vf_config=self.config.vf
            )
            # Set the encoder to the policy's networks' encoder.
            self.non_episodic_vf.set_encoder(self.rl_modules["default_policy"].encoder)            
            # Add the parameters to the policy's networks' optimizer.
            
            
            self.gamma = self.config.gamma
            self.lambda_ = self.config.lambda_
            self.vf_loss_coeff = self.config.vf_loss_coeff
            self.adv_int_coeff = self.config.adv_int_coeff
            self.adv_ext_coeff = self.config.adv_ext_coeff
        
        self.normalize = self.config.normalize
        # Use a standardization of intrinsic rewards.
        self._moving_mean_std = None
        self.sub_exploration = self.config.get("sub_exploration", None)
        if self.sub_exploration is None:
            # Define Stochastic Sampling here.
            sub_exploration = {}
        
    def on_make_optimizer(self):
        """Create an optimizer for the distill_net"""
        config = self.config.optimizer_config
        self.optimizer = torch.optim.Adam(self.distill_predictor_net.parameters(), lr=config.lr)
        
        # If nonepisodic returns are calculated add the loss
        # to the loss of the RLTrainer loss.
        if self.non_episodic_returns:
            """Here it would be nice to have the possibility to use the same optimizer as for Pi/VF"""
            self.vf_optimizer = torch.optim.Adam(self.nonepisodic_vf.parameters(), lr=self.lr)
            
    def on_episode_end(self, train_batch: SampleBatch):
        """Calculate the advantages and intrinsic rewards"""
        intrinsic_rewards = self._compute_intrinsic_rewards(train_batch, ...)
        if self.non_episodic_returns:
            # Calculate the advantages
            vf = self.nonepisodic_vf(batch)
            compute_advantages(vf, train_batch, self.gamma, self.lambda_, ...)
        else: 
            # Add intrinsic rewards to extrinsic rewards
            train_batch["REWARDS"] += intrinsic_rewards
            
        # Also nice, if the distill_net could be trained here, if needed in 
        # higher frequency.
        if not self.global_update:
            self._distill_net_update(train_batch)
        
    def on_after_loss(self, train_batch: SampleBatch, loss_out: Dict[LossID, LossTensor], ...):
        """Calculate the nonepisodic VF loss"""
        
        if self.nonepisodic_returns:
            # Compute the nonepisodic VF loss
            self.vf_intrinsic_loss = self._compute_nonepisodic_value_loss(train_batch)
            loss_out.update({
                "exploration_loss": self.vf_intrinsic_loss,
            })
            # Then the nonepisodic VF network has to be optimized by the RLTrainer.
            return loss_out
