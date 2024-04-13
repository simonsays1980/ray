import gymnasium as gym
from typing import Any, Dict
from ray.rllib.algorithms.dreamerv3.utils import (
    get_gru_units,
    get_num_z_categoricals,
    get_num_z_classes,
)
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override


class DreamerV3Catalog(Catalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: Dict[ystr, Any],
    ):

        # Initialize the Catalog class.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        # This can be a pre-configured or user-configured config.
        # NOTE: Can be `None`, if user-configured.
        self.model_size: str = self._model_config_dict.get("model_size")
        # Compute the size of the vector coming out of the sequence model.
        # TODO (simon): Retrieve from sequence model itself.
        self.h_plus_z_flat = get_gru_units(self.model_size) + (
            get_num_z_categoricals(self.model_size) * get_num_z_classes(self.model_size)
        )

    def build_decoder(self, framework: str) -> Model:
        """Builds the World-Model's decoder network depending on the obs space."""
        return self._get_decoder_config().build(framework=framework)
    
    def build_sequence_model(self, framework: str) -> Model:
        """Builds the World-Model's sequence model network."""
        return self._get_sequence_model_config().build(framework=framework)

    @override(Catalog)
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: Dict[str, Any],
        action_space: gym.Space,
        view_requirements: Dict[str, Any],
    ) -> ModelConfig:
        """Defines the configuration to be used for the encoder.

        Note, this overrides the default method and gets called automatically.
        """

        # Only update 'model_config_dict' if not user-defined.
        model_size = model_config_dict.get("model_size")
        encoder_model_config_dict = model_config_dict.get("encoder_model_config_dict")
        if model_size:
            # Uses pre-configured models to pass to default encoders.
            encoder_model_config_dict = cls.get_encoder_model_config_dict(
                model_config_dict
            )

        # Return the `Catalog`'s method to enable automatic build.
        return super()._get_encoder_config(
            observation_space,
            encoder_model_config_dict,
            action_space,
            view_requirements,
        )

    def _get_decoder_config(self):
        decoder_model_config_dict = self._model_config_dict.get(
            "decoder_model_config_dict"
        )

        if self.model_size:
            decoder_model_config_dict = self.get_decoder_model_config_dict(
                self._model_config_dict
            )

        # If we decode an image space we use transpose convolution.
        if self.is_img_space:
            # Use our basic CNNTransposeHeadConfig.
            from ray.rllib.core.models.configs import CNNTransposeHeadConfig

            # TODO (simon): Rename the model config keys.
            return CNNTransposeHeadConfig(
                input_dims=[self.h_plus_z_flat],
                initial_image_dims=decoder_model_config_dict["initial_image_dims"],
                cnn_transpose_filter_specifiers=decoder_model_config_dict[
                    "conv_filters"
                ],
                # TODO (simon): Implement these keys in the catalog.
                cnn_transpose_use_bias=decoder_model_config_dict[
                    "conv_transpose_use_bias"
                ],
                cnn_transpose_use_layernorm=decoder_model_config_dict[
                    "conv_transpose_use_layernorm"
                ],
                cnn_transpose_activation=decoder_model_config_dict["conv_activation"],
                cnn_transpose_kernel_initializer=decoder_model_config_dict[
                    "conv_transpose_kernel_initializer"
                ],
                cnn_transpose_kernel_initializer_config=decoder_model_config_dict[
                    "conv_transpose_kernel_initializer_config"
                ],
                cnn_transpose_bias_initializer=decoder_model_config_dict[
                    "conv_transpose_bias_initializer"
                ],
                cnn_transpose_bias_initializer_config=decoder_model_config_dict[
                    "conv_transpose_bias_initializer_config"
                ],
            )
        # Otherwise we fall back to a plain MLP.
        else:
            from ray.rllib.core.models.configs import MLPHeadConfig

            return MLPHeadConfig(
                input_dims=[self.h_plus_z_flat],
                hidden_layer_dims=decoder_model_config_dict["post_fcnet_hiddens"],
                hidden_layer_activation=decoder_model_config_dict[
                    "post_fcnet_activation"
                ],
                # TODO (simon): Not yet available.
                # hidden_layer_use_layernorm=self.decoder_model_config_dict[
                #     "hidden_layer_use_layernorm"
                # ],
                # hidden_layer_use_bias=self.decoder_model_config_dict["hidden_layer_use_bias"],
                hidden_layer_weights_initializer=decoder_model_config_dict[
                    "post_fcnet_weights_initializer"
                ],
                hidden_layer_weights_initializer_config=decoder_model_config_dict[
                    "post_fcnet_weights_initializer_config"
                ],
                hidden_layer_bias_initializer=decoder_model_config_dict[
                    "post_fcnet_bias_initializer"
                ],
                hidden_layer_bias_initializer_config=decoder_model_config_dict[
                    "post_fcnet_bias_initializer_config"
                ],
                output_layer_activation="linear",
                output_layer_dim=self.observation_space.shape[0],
                # TODO (simon): Not yet available.
                # output_layer_use_bias=self._model_config_dict["output_layer_use_bias"],
                output_layer_weights_initializer=decoder_model_config_dict[
                    "post_fcnet_weights_initializer"
                ],
                output_layer_weights_initializer_config=decoder_model_config_dict[
                    "post_fcnet_weights_initializer_config"
                ],
                output_layer_bias_initializer=decoder_model_config_dict[
                    "post_fcnet_bias_initializer"
                ],
                output_layer_bias_initializer_config=decoder_model_config_dict[
                    "post_fcnet_bias_initializer_config"
                ],
            )

    def _get_sequence_model_config(self) -> ModelConfig:
        """Returns the sequence model configuration."""

        sequence_model_config_dict = self._model_config_dict.get(
            "sequence_model_config_dict"
        )
        if self.model_size:
            sequence_model_config_dict = self.get_sequence_model_config_dict(
                sequence_model_config_dict
            )

        uses_tokenizer = self._model_config_dict.get(
            "tokenizer_model_config_dict", False
        )
        if uses_tokenizer:
            tokenizer_config = self._get_tokenizer_config()
        # TODO (simon): Maybe offering an LSTM in addition to GRU?
        from ray.rllib.core.models.configs import RecurrentEncoderConfig

        return RecurrentEncoderConfig(
            input_dims=tokenizer_config.output_layer_dim
            if uses_tokenizer
            else [int(self.h_plus_z_flat + self.a_flat)],
            recurrent_layer_type="gru",
            hidden_dim=sequence_model_config_dict["gru_cell_size"],
            num_layers=sequence_model_config_dict["gru_num_layers"],
            time_major=True,
            hidden_weights_initializer=sequence_model_config_dict[
                "hidden_weights_initializer"
            ],
            # In the original DreamerV3 code, there is a prior MLP layer.
            # We use instead for this the tokenizer of a recurrent encoder.
            encoder_config=self._get_tokenizer_config() if uses_tokenizer else None,
        )

    def _get_tokenizer_config(self) -> ModelConfig:
        """Returns the tokenizer configuration."""
        # TODO (simon): Implement this method.
        # Returns the config for the tolenizer model in the sequence model.
        tokenizer_model_config_dict = self._model_config_dict.get(
            "tokenizer_model_config_dict"
        )
        if self.model_size:
            tokenizer_model_config_dict = self.get_tokenizer_model_config_dict(
                tokenizer_model_config_dict
            )

        fcnet_hiddens = tokenizer_model_config_dict["fcnet_hiddens"]
        # TODO (sven): Move to a new ModelConfig object (dataclass) asap, instead of
        #  "linking" into the old ModelConfig (dict)! This just causes confusion as to
        #  which old keys now mean what for the new RLModules-based default models.
        encoder_latent_dim = (
            tokenizer_model_config_dict["encoder_latent_dim"] or fcnet_hiddens[-1]
        )
        if tokenizer_model_config_dict["encoder_latent_dim"]:
            hidden_layer_dims = tokenizer_model_config_dict["fcnet_hiddens"]
        else:
            hidden_layer_dims = tokenizer_model_config_dict["fcnet_hiddens"][:-1]
        from ray.rllib.core.models.configs import MLPEncoderConfig

        return MLPEncoderConfig(
            input_dims=[int(self.h_plus_z_flat + self.a_flat)],
            hidden_layer_dims=hidden_layer_dims,
            hidden_layer_activation=tokenizer_model_config_dict["fcnet_activation"],
            # TODO (simon): Not yet available.
            # hidden_layer_use_layernorm=tokenizer_model_config_dict[
            #     "hidden_layer_use_layernorm"
            # ],
            # hidden_layer_use_bias=tokenizer_model_config_dict["hidden_layer_use_bias"],
            hidden_layer_weights_initializer=tokenizer_model_config_dict[
                "fcnet_weights_initializer"
            ],
            hidden_layer_weights_initializer_config=tokenizer_model_config_dict[
                "fcnet_weights_initializer_config"
            ],
            hidden_layer_bias_initializer=tokenizer_model_config_dict[
                "fcnet_bias_initializer"
            ],
            hidden_layer_bias_initializer_config=tokenizer_model_config_dict[
                "fcnet_bias_initializer_config"
            ],
            output_layer_activation=tokenizer_model_config_dict[
                "post_fcnet_activation"
            ],
            output_layer_dim=encoder_latent_dim,
            # TODO (simon): Not yet available.
            # output_layer_use_bias=self._model_config_dict["output_layer_use_bias"],
            output_layer_weights_initializer=tokenizer_model_config_dict[
                "post_fcnet_weights_initializer"
            ],
            output_layer_weights_initializer_config=tokenizer_model_config_dict[
                "post_fcnet_weights_initializer_config"
            ],
            output_layer_bias_initializer=tokenizer_model_config_dict[
                "post_fcnet_bias_initializer"
            ],
            output_layer_bias_initializer_config=tokenizer_model_config_dict[
                "post_fcnet_bias_initializer_config"
            ],
        )
    
    def _get_dynamics_predictor_config(self) -> ModelConfig:
        """Returns the dynamics predictor configuration."""
        
        dynamics_predictor_model_config_dict = self._model_config_dict.get(
            "dynamics_predictor_model_config_dict"
        )
        if self.model_size:
            dynamics_predictor_model_config_dict = self.get_dynamics_predictor_model_config_dict(
                dynamics_predictor_model_config_dict
            )
        
        from ray.rllib.algorithms.dreamerv3.utils.configs import DynamicsPredictorConfig

        return DynamicsPredictorConfig(
            # TODO (simon): Rename to`h_flat`.
            input_dims=self.num_gru_units,
            hidden_layer_dims=dynamics_predictor_model_config_dict["fcnet_hiddens"],
        )

    def get_encoder_model_config_dict(
        cls, model_config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Returns the encoder model config dictionary."""
        # TODO (simon): Implement this method.
        # Returns the config for model-size model.
        pass

    def get_sequence_model_config_dict(
        cls, model_config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Returns the sequence model config dictionary."""
        # TODO (simon): Implement this method.
        # Returns the config for model-size model.
        pass

    def get_tokenizer_model_config_dict(
        cls, model_config_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Returns the tokenizer model config dictionary."""
        # TODO (simon): Implement this method.
        # Returns the config for model-size model.
        pass
