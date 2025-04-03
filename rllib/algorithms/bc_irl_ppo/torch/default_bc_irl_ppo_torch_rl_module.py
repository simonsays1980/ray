from ray.rllib.algorithms.bc_irl_ppo.bc_irl_ppo_catalog import BCIRLPPOCatalog
from ray.rllib.algorithms.bc_irl_ppo.default_bc_irl_ppo_rl_module import (
    DefaultBCIRLRewardRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class DefaultBCIRLRewardTorchRLModule(TorchRLModule, DefaultBCIRLRewardRLModule):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = BCIRLPPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def _forward(self, batch):
        inputs = {}
        output = {}

        actions = nn.functional.one_hot(
            batch[Columns.ACTIONS].long(), self.action_space.n
        ).float()
        inputs[Columns.OBS] = torch.concatenate(
            [batch[Columns.OBS], actions, batch[Columns.NEXT_OBS]], dim=-1
        )
        encoder_outs = self.rf_encoder(inputs)

        output[Columns.REWARDS] = self.rf(encoder_outs[ENCODER_OUT]).squeeze(dim=-1)
        return output
