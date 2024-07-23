import numpy as np
import os
from pathlib import Path
import unittest

import ray
import ray.rllib.algorithms.marwil as marwil
from ray.rllib.algorithms.marwil.marwil_tf_policy import MARWILTF2Policy
from ray.rllib.algorithms.marwil.marwil_torch_policy import MARWILTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.offline import JsonReader
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)
from ray.rllib.utils.test_utils import (
    check,
    check_compute_single_action,
    check_train_results,
    framework_iterator,
)

torch, _ = try_import_torch()


class TestMARWIL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_marwil_compilation_and_learning_from_offline_file(self):
        """Test whether a MARWILAlgorithm can be built with all frameworks.

        Learns from a historic-data file.
        To generate this data, first run:
        $ ./train.py --run=PPO --env=CartPole-v1 \
          --stop='{"timesteps_total": 50000}' \
          --config='{"output": "/tmp/out", "batch_mode": "complete_episodes"}'
        """
        data_path = "tests/data/cartpole/cartpole-v1_large"
        base_path = Path(__file__).parents[3]
        print(f"base_path={base_path}")
        data_path = "local://" + base_path.joinpath(data_path).as_posix()
        print(f"data_path={data_path}")

        config = (
            marwil.MARWILConfig()
            .environment(env="CartPole-v1")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .evaluation(
                evaluation_interval=3,
                evaluation_num_env_runners=1,
                evaluation_duration=5,
                evaluation_parallel_to_training=True,
            )
            .offline_data(input_=[data_path])
            .training(
                lr=0.0008,
                train_batch_size_per_learner=2000,
            )
        )

        num_iterations = 350
        min_reward = 100.0

        # Test for all frameworks.
        
        algo = config.build()
        learnt = False
        for i in range(num_iterations):
            results = algo.train()
            check_train_results(results)
            print(results)

            eval_results = results.get(EVALUATION_RESULTS, {})
            if eval_results:
                episode_return_mean = eval_results[ENV_RUNNER_RESULTS][
                    EPISODE_RETURN_MEAN
                ]
                print(f"iter={i}, R={episode_return_mean}")
                # Learn until some reward is reached on an actual live env.
                if episode_return_mean > min_reward:
                    print("BC has learnt the task!")
                    learnt = True
                    break

        if not learnt:
            raise ValueError(
                f"`MARWIL` did not reach {min_reward} reward from expert offline data!"
            )

        check_compute_single_action(algo, include_prev_action_reward=True)

        algo.stop()

    # def test_marwil_cont_actions_from_offline_file(self):
    #     """Test whether MARWIL runs with cont. actions.

    #     Learns from a historic-data file.
    #     To generate this data, first run:
    #     $ ./train.py --run=PPO --env=Pendulum-v1 \
    #       --stop='{"timesteps_total": 50000}' \
    #       --config='{"output": "/tmp/out", "batch_mode": "complete_episodes"}'
    #     """
    #     rllib_dir = Path(__file__).parent.parent.parent.parent
    #     print("rllib dir={}".format(rllib_dir))
    #     data_file = os.path.join(rllib_dir, "tests/data/pendulum/large.json")
    #     print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

    #     config = (
    #         marwil.MARWILConfig()
    #         .env_runners(num_env_runners=1)
    #         .evaluation(
    #             evaluation_num_env_runners=1,
    #             evaluation_interval=3,
    #             evaluation_duration=5,
    #             evaluation_parallel_to_training=True,
    #             # Evaluate on actual environment.
    #             evaluation_config=marwil.MARWILConfig.overrides(input_="sampler"),
    #             off_policy_estimation_methods={},
    #         )
    #         .offline_data(
    #             # Learn from offline data.
    #             input_=[data_file],
    #         )
    #     )

    #     num_iterations = 3

    #     # Test for all frameworks.
    #     for _ in framework_iterator(config, frameworks=("torch", "tf")):
    #         algo = config.build(env="Pendulum-v1")
    #         for i in range(num_iterations):
    #             print(algo.train())
    #         algo.stop()

    # def test_marwil_loss_function(self):
    #     """
    #     To generate the historic data used in this test case, first run:
    #     $ ./train.py --run=PPO --env=CartPole-v1 \
    #       --stop='{"timesteps_total": 50000}' \
    #       --config='{"output": "/tmp/out", "batch_mode": "complete_episodes"}'
    #     """
    #     rllib_dir = Path(__file__).parent.parent.parent.parent
    #     print("rllib dir={}".format(rllib_dir))
    #     data_file = os.path.join(rllib_dir, "tests/data/cartpole/small.json")
    #     print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

    #     config = (
    #         marwil.MARWILConfig()
    #         .env_runners(num_env_runners=0)
    #         .offline_data(input_=[data_file])
    #     )  # Learn from offline data.

    #     for fw, sess in framework_iterator(config, session=True):
    #         reader = JsonReader(inputs=[data_file])
    #         batch = reader.next()

    #         algo = config.build(env="CartPole-v1")
    #         policy = algo.get_policy()
    #         model = policy.model

    #         # Calculate our own expected values (to then compare against the
    #         # agent's loss output).
    #         cummulative_rewards = compute_advantages(
    #             batch, 0.0, config.gamma, 1.0, False, False
    #         )["advantages"]
    #         if fw == "torch":
    #             cummulative_rewards = torch.tensor(cummulative_rewards)
    #         if fw != "tf":
    #             batch = policy._lazy_tensor_dict(batch)
    #         model_out, _ = model(batch)
    #         vf_estimates = model.value_function()
    #         if fw == "tf":
    #             model_out, vf_estimates = policy.get_session().run(
    #                 [model_out, vf_estimates]
    #             )
    #         adv = cummulative_rewards - vf_estimates
    #         if fw == "torch":
    #             adv = adv.detach().cpu().numpy()
    #         adv_squared = np.mean(np.square(adv))
    #         c_2 = 100.0 + 1e-8 * (adv_squared - 100.0)
    #         c = np.sqrt(c_2)
    #         exp_advs = np.exp(config.beta * (adv / c))
    #         dist = policy.dist_class(model_out, model)
    #         logp = dist.logp(batch["actions"])
    #         if fw == "torch":
    #             logp = logp.detach().cpu().numpy()
    #         elif fw == "tf":
    #             logp = sess.run(logp)
    #         # Calculate all expected loss components.
    #         expected_vf_loss = 0.5 * adv_squared
    #         expected_pol_loss = -1.0 * np.mean(exp_advs * logp)
    #         expected_loss = expected_pol_loss + config.vf_coeff * expected_vf_loss

    #         # Calculate the algorithm's loss (to check against our own
    #         # calculation above).
    #         batch.set_get_interceptor(None)
    #         postprocessed_batch = policy.postprocess_trajectory(batch)
    #         loss_func = (
    #             MARWILTF2Policy.loss if fw != "torch" else MARWILTorchPolicy.loss
    #         )
    #         if fw != "tf":
    #             policy._lazy_tensor_dict(postprocessed_batch)
    #             loss_out = loss_func(
    #                 policy, model, policy.dist_class, postprocessed_batch
    #             )
    #         else:
    #             loss_out, v_loss, p_loss = policy.get_session().run(
    #                 # policy._loss is create by TFPolicy, and is basically the
    #                 # loss tensor of the static graph.
    #                 [
    #                     policy._loss,
    #                     policy._marwil_loss.v_loss,
    #                     policy._marwil_loss.p_loss,
    #                 ],
    #                 feed_dict=policy._get_loss_inputs_dict(
    #                     postprocessed_batch, shuffle=False
    #                 ),
    #             )

    #         # Check all components.
    #         if fw == "torch":
    #             check(policy.v_loss, expected_vf_loss, decimals=4)
    #             check(policy.p_loss, expected_pol_loss, decimals=4)
    #         elif fw == "tf":
    #             check(v_loss, expected_vf_loss, decimals=4)
    #             check(p_loss, expected_pol_loss, decimals=4)
    #         else:
    #             check(policy._marwil_loss.v_loss, expected_vf_loss, decimals=4)
    #             check(policy._marwil_loss.p_loss, expected_pol_loss, decimals=4)
    #         check(loss_out, expected_loss, decimals=3)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
