def check_if_diag_gaussian(action_distribution_cls, framework):
    """Checks, if `free_log_std` can be used.
    
    This is only supported for an DiagGaussian heritage in this library.
    """
    if framework == "torch":
        from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

        assert issubclass(action_distribution_cls, TorchDiagGaussian), (
            f"free_log_std is only supported for DiagGaussian action distributions. "
            f"Found action distribution: {action_distribution_cls}."
        )
    elif framework == "tf2":
        from ray.rllib.models.tf.tf_distributions import TfDiagGaussian

        assert issubclass(action_distribution_cls, TfDiagGaussian), (
            "free_log_std is only supported for DiagGaussian action distributions. "
            "Found action distribution: {}.".format(action_distribution_cls)
        )
    else:
        raise ValueError(f"Framework {framework} not supported for free_log_std.")