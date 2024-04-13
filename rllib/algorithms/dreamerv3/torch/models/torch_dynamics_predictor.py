from typing import Any, Dict

from ray.rllib.algorithms.dreamerv3.utils.configs import DynamicsPredictorConfig
from ray.rllib.algorithms.dreamerv3.torch.models.torch_representation_layer import (
    TorchRepresentationLayer,
)
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.utils.annotations import override

# TODO (simon): Redefine globally for DreamerV3.
REPRESENTATIONS = "z"
Z_PROBS = "z_probs"


class TorchDynamicsPredictor(TorchModel):
    def __init__(self, config: DynamicsPredictorConfig):
        # Intializer the parent class.
        super().__init__(config)

        # This is neither an encoder nor a head, so we use the primitive here.
        self.mlp = TorchMLP(
            input_dim=config.input_dims[0],
            hidden_layer_dims=config.hidden_layer_dims,
            hidden_layer_activation=config.hidden_layer_activation,
            hidden_layer_use_bias=config.hidden_layer_use_bias,
            hidden_layer_use_layernorm=config.hidden_layer_use_layernorm,
            hidden_layer_weights_initializer=config.hidden_layer_weights_initializer,
            hidden_layer_weights_initializer_config=config.hidden_layer_weights_initializer_config,
            hidden_layer_bias_initializer=config.hidden_layer_bias_initializer,
            hidden_layer_bias_initializer_config=config.hidden_layer_bias_initializer_config,
            # Note, no output layer, i.e. last hidden layer defines the output size.
            output_layer_dim=None,
        )

        self.representation_layer = TorchRepresentationLayer(
            input_size=config.hidden_layer_dims[-1],
            num_categoricals=config.num_categoricals,
            num_classes_per_categorical=config.num_classes_per_categorical,
        )

    @override(TorchModel)
    def forward(self, batch: Dict[str, Any], return_z_probs: bool = False):
        """Forward pass through the dynamics predictor."""

        out = {}
        mlp_outs = self.mlp(batch)
        if return_z_probs:
            out[REPRESENTATIONS], out[Z_PROBS] = self.representation_layer(
                mlp_outs, return_z_probs
            )
        else:
            out[REPRESENTATIONS] = self.representation_layer(mlp_outs, return_z_probs)
        return out
