from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch.nn import nn
from torch.optim import Optimizer

from ray.rllib2.algorithms.callbacks import (
    CallbacksConfig,
    RLlibCallbacks
)
from ray.rllib2.models.configs import ModelConfig
from ray.rllib2.models.torch.encoder import ModelWithEncoder

LossID = str
BatchType = "BatchType"
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
    optimizer_config: Dict = None
    # Learning rate for the distillation optimizer.
    lr: float = 1e-5
    # Intrinsic reward coefficient.
    intrinsic_reward_coeff: float = 5e-3
    # If the distillation model should be synchronously.
    synchronous_update: bool = False
    # If a nonepisodic value head should be used.
    non_episodic_returns: bool = False
    gamma: float = 0.99
    lambda_: float = 0.95
    vf_loss_coeff: float = 0.4
    adv_int_coeff: float = 1.0
    adv_ext_coeff: float = 2.0
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
        self.synchronous_update = self.config.get("synchronous_update")
        
        # @kourosh: Here it would be good to make the encoder of Pi, QF, 
        # or VF networks available. So parameter sharing could be used in 
        # the `distill_net` (not the distill_target_net).
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
        
        # @kourosh: This might not be needed, if we consider exploration to take part 
        # in multiple callbacks: One for Subexploration via Stochastic Sampling choosing
        # the action. Another for exploration via curiosity or RND. 
        self.sub_exploration = self.config.get("sub_exploration", None)
        if self.sub_exploration is None:
            # Define Stochastic Sampling here.
            sub_exploration = {}
        
    def on_make_optimizer(self) -> Dict[LossID, Optimizer]:
        """
        Create an optimizer for the distillation net. If nonepisodic
        returns (a second value head) is used add parameters to the existent 
        optimizer from Pi, VF, or QF.
        If the optimizing of the distillation network is done locally do not pass 
        it over to the RLTrainer.
        """
        config = self.config.optimizer_config      
        optimizer_dict = {}
        optimizer = torch.optim.Adam(self.distill_predictor_net.parameters(), lr=config.lr)
        if not self.synchronous_update:  
            # Update the model locally for each RolloutWorker.
            # @kourosh: Do we still have access to a device as before self.device was existent for 
            # the exploration class?
            # self.distill_target_model.to(self.device)
            self.optimizer = optimizer
        else:
            # Update globally and synch the weights to the remote workers.
            # @kourosh: What do you think about naming. More specific - like here - or more
            # genral like "exploration_loss"?
            optimizer_dict.update({"exploration_rnd_distillation_loss": optimizer})
        
        # If nonepisodic returns are calculated add the loss
        # to the loss of the RLTrainer loss.
        if self.non_episodic_returns:
            # @kourosh: Here it would be nice to have the opportunity to use the optimizer of 
            # Pi/VF from the RLTrainer and simply add parameters and loss to it.                    
            optimizer_dict.update({
                "exploration_rnd_intrinsic_vf_loss": torch.optim.Adam(self.nonepisodic_vf.parameters(), lr=self.lr)
            })
        return optimizer_dict
            
    # @kourosh: Is on postprocess_trajectory vanishing? I do remember sth you said on the summit.            
    def on_postprocess_trajectory(self, train_batch: SampleBatch):
        """Calculate the advantages and intrinsic rewards"""
        intrinsic_rewards = self._compute_intrinsic_rewards(train_batch, ...)
        if self.non_episodic_returns:
            # Calculate the values.
            vf = self.nonepisodic_vf(train_batch)
            # In-place operation, computes advantages and value targets.
            compute_advantages(vf, train_batch, intrinsic_rewards, self.gamma, self.lambda_, ...)
        else: 
            # Add intrinsic rewards to extrinsic rewards
            train_batch["REWARDS"] += intrinsic_rewards
            
        # Also nice, if the distill_net could be trained here, if needed in 
        # higher frequency.
        if not self.synchronous_update:
            # Update the distillation network locally on each RolloutWorker.
            loss_out = self._compute_loss(train_batch)
            self._compute_grads_and_apply_if_needed(
                batch=train_batch,
                loss_out=loss_out,
                apply_grad=True
            )
            
        
    def on_after_loss(self, train_batch: SampleBatch, loss_out: Dict[LossID, LossTensor], ...):
        """Calculate the nonepisodic VF loss"""
        
        if self.synchronous_update:
            loss_out.update({
                "exploration_rnd_distillation_loss": self._compute_loss(train_batch)
            })
        if self.nonepisodic_returns:
            # Compute the nonepisodic VF loss
             # Then the nonepisodic VF network has to be optimized by the RLTrainer.            
            loss_out.update({
                "exploration_rnd_nonepisodic_vf_loss": self._compute_non_episodic_value_loss(train_batch),
            })
        return loss_out     
        
          
    def _compute_loss(self, train_batch: SampleBatch) -> Dict[LossID, LossTensor]:       
        novelty = self._compute_novelty(train_batch)
        distill_loss = torch.mean(novelty)            
        return {
            "exploration_rnd_distillation_loss": distill_loss
        }
        
        
    def _compute_non_episodic_value_loss(self, train_batch: SampleBatch) -> Dict[LossID, LossTensor]:
        self.vf_intrinsic_loss = (
            torch.mean(
                torch.pow(
                    train_batch["exploration_vf_preds"] 
                    - train_batch["exploration_value_targets"], 2.0,
                )
            ) * self.vf_loss_coeff
        )
        return {"exploration_rnd_intrinsic_vf_loss": self.vf_intrinsic_loss}


    def _compute_intrinsic_rewards(self, train_batch):
        """Computes the intrinsic reward."""
        self.intrinsic_reward = self.novelty * self.intrinsic_reward_coeff
        
        
    def _compute_novelty(self, train_batch: SampleBatch):
        phi = self.distill_net.forward(train_batch[SampleBatch.OBS])
        phi_target = self.distill_target_net.forward(train_batch[SampleBatch.OBS])
        # Avoid dividing by zero int he gradient by adding a small slack variable.
        self.novelty = torch.norm(phi - phi_target + 1e-12, dim=1)
        return self.novelty
    
    
    def _compute_grads_and_apply_if_needed(
        self, 
        batch: BatchType,
        fwd_out, 
        loss_out: Dict[LossID, torch.Tensor],
        apply_grad: bool = True,
    ) -> Any:
        """This updates the distillation network in asynchronous mode."""
        # This optimizer is local to each RolloutWorker.
        self.optimizer.zero_grad()
        # Only use the distillation loss, if more losses exist.
        loss_out["exploration_rnd_distillation_loss"].backward()
        
        if apply_grad:
            self.optimizer.step()      
            
            
    """Needs sync_weights() function for synchronous updates."""