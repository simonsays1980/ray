"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
if torch:
    F = nn.functional


class RewardPredictorLayer(nn.Module):
    """A layer outputting reward predictions using K bins and two-hot encoding.

    This layer is used in two models in DreamerV3: The reward predictor of the world
    model and the value function. K is 255 by default (see [1]) and doesn't change
    with the model size.

    Possible predicted reward/values range from symexp(-20.0) to symexp(20.0), which
    should cover any possible environment. Outputs of this layer are generated by
    generating logits/probs via a single linear layer, then interpreting the probs
    as weights for a weighted average of the different possible reward (binned) values.
    """

    def __init__(
        self,
        *,
        input_size: int,
        num_buckets: int = 255,
        lower_bound: float = -20.0,
        upper_bound: float = 20.0,
    ):
        """Initializes a RewardPredictorLayer instance.

        Args:
            input_size: The input size of the reward predictor layer.
            num_buckets: The number of buckets to create. Note that the number of
                possible symlog'd outcomes from the used distribution is
                `num_buckets` + 1:
                lower_bound --bucket-- o[1] --bucket-- o[2] ... --bucket-- upper_bound
                o=outcomes
                lower_bound=o[0]
                upper_bound=o[num_buckets]
            lower_bound: The symlog'd lower bound for a possible reward value.
                Note that a value of -20.0 here already allows individual (actual env)
                rewards to be as low as -400M. Buckets will be created between
                `lower_bound` and `upper_bound`.
            upper_bound: The symlog'd upper bound for a possible reward value.
                Note that a value of +20.0 here already allows individual (actual env)
                rewards to be as high as 400M. Buckets will be created between
                `lower_bound` and `upper_bound`.
        """
        self.num_buckets = num_buckets
        super().__init__()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.reward_buckets_layer = nn.Linear(
            in_features=input_size,
            out_features=self.num_buckets,
            bias=True
        )
        self.reward_buckets_layer.weight.data.fill_(0.0)
        self.reward_buckets_layer.bias.data.fill_(0.0)

    def forward(self, inputs, return_logits=False):
        """Computes the expected reward using N equal sized buckets of possible values.

        Args:
            inputs: The input tensor for the layer, which computes the reward bucket
                weights (logits). [B, dim].
            return_logits: Whether to return the logits over the reward buckets
                as a second return value (besides the expected reward).

        Returns:
            The expected reward OR a tuple consisting of the expected reward and the
            tfp `FiniteDiscrete` distribution object. To get the individual bucket
            probs, do `[FiniteDiscrete object].probs`.
        """
        # Compute the `num_buckets` weights.
        logits = self.reward_buckets_layer(inputs)

        # Compute the expected(!) reward using the formula:
        # `softmax(Linear(x))` [vectordot] `possible_outcomes`, where
        # `possible_outcomes` is the even-spaced (binned) encoding of all possible
        # symexp'd reward/values.
        probs = F.softmax(logits, dim=-1)
        possible_outcomes = torch.linspace(
            self.lower_bound,
            self.upper_bound,
            self.num_buckets,
            device=logits.device
        )

        # Simple vector dot product (over last dim) to get the mean reward
        # weighted sum, where all weights sum to 1.0.
        expected_rewards = torch.sum(probs * possible_outcomes, dim=-1)

        if return_logits:
            return expected_rewards, logits
        return expected_rewards
