import torch
from typing import Dict
from ray.rllib2.algorithms.callbacks import RLlibCallbacks

class RNDExplorationCallbacks(RLlibCallbacks):
    
    def __init__(self, config: CallbackConfig):
        # Initialize super.
        super().__init__(config=config)
        self.config = config
        self.distill_net_config = config.get("distill_net_config", None)
        if not self.distill_net_config:
            self.distill_net_config = config.get("model_config").copy()
        
        self.embed_dim = config.get("embed_dim")
        self.lr = config.get("lr")
        
        self.distill_predictor_net = model_catalog.make_mlp(
            obs_space=self.config.obs_space,
            embed_dim=self.embed_dim,
            mlp_config=self.distill_net_config,
        )
        
        self.distill_target_net = model_catalog.make_mlp(
            obs_space=self.config.obs_space,
            embed_dim=self.embed_dim,
            mlp_config=self.distill_net_config,
        )
        
        self.non_episodic_returns = self.config.non_episodic_returns
        if self.non_episodic_returns:
            """Get access to the encoder of the policy!!!"""
            self.encoder = self.rl_modules["agent_module"].encoder
            self.nonepisodic_vf = model_catalog.make_vf(
                obs_space=self.config.obs_space,
                action_space=self.config.action_space,
                vf_config=self.config.vf
            )
            
            self.gamma = self.config.gamma
            self.lambda_ = self.config.lambda_
            """Use also here the policies vf_loss_coeff"""
            self.vf_loss_coeff = self.config.vf_loss_coeff
            self.adv_int_coeff = self.config.adv_int_coeff
            self.adv_ext_coeff = self.config.adv_ext_coeff
        
        self.normalize = self.config.normalize
        # Use a standardization of intrinsic rewards.
        self._moving_mean_std = None
        self.sub_exploration = self.config.get("sub_exploration", None)
        if self.sub_exploration is None:
            sub_exploration = {}
        
    def on_make_optimizer(self):
        """Create an optimizer for the distill_net"""
        config = self.config.optimizer_config
        self.optimizer = torch.optim.Adam(self.distill_predictor_net.parameters(), lr=self.lr)
        
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
        
        
            
            
            
        
        
        
    
            
    

        
        
        
        
        
            
            
            
        
        
        
        
        