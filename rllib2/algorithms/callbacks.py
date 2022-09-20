
class RLlibCallbacks:
    
    def __init__(self, config: CallbackConfig):
        """
        The CallbackConfig should be used to transport infos from the main config
        to the callback, e.g. model_config, etc.
        """
    # for supporting exploration
    def on_postprocess_trajectory(self, ...):
        """
        This can be used for adding intrinsic reward to the extrinsic reward.
        """
        pass

    def on_after_forward(self, ...):
        """
        This can be used for modifying the output distribution of the policy for sampling actions from the env.
        # can implement epsilon greedy
        """
        pass

    def on_episode_start(self, ...):
        """
        This can be used to add noise to the RLModule parameters that samples actions
        """
        pass
    
    def on_episode_end(self, ...):
        pass
    
    def on_after_loss(self, ...):
        """
        This can augment the loss dict returned by the UnitTrainer.loss() to augment the loss with specific losses from Exploration
        """
        pass
        
    def on_before_loss(self, ...):
        """
        Preparation before calling UnitTrainer.loss() 
        """
        pass
    
    def on_make_optimizer(self, ...):
        """
        After calling make_optimizer inside the init of the UnitTrainer
        """
        pass