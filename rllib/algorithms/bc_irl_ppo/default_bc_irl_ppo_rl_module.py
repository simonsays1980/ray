from ray.rllib.core.columns import Columns
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module import MultiRLModule, RLModule
from ray.rllib.utils.annotations import override


class DefaultBCIRLRewardRLModule(RLModule):
    @override(RLModule)
    def setup(self):

        self.rf_encoder = self.catalog.build_rf_encoder(framework=self.framework)

        self.rf = self.catalog.build_rf_head(framework=self.framework)

        @override(RLModule)
        def get_initial_state(self) -> dict:
            return {}

        @override(RLModule)
        def input_specs_train(self) -> SpecType:
            return [
                Columns.OBS,
                Columns.ACTIONS,
                Columns.NEXT_OBS,
            ]

        @override(RLModule)
        def output_specs_train(self) -> SpecType:
            return [
                Columns.REWARDS,
            ]
