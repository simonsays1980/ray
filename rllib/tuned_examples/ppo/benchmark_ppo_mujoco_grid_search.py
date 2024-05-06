import re
import time
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.connectors.env_to_module.mean_std_filter import MeanStdFilter
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray import train, tune

# Needs the following packages to be installed on Ubuntu:
#   sudo apt-get libosmesa-dev
#   sudo apt-get install patchelf
#   python -m pip install "gymnasium[mujoco]"
# Might need to be added to bashsrc:s
#   export MUJOCO_GL=osmesa"
#   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"

parser = add_rllib_example_script_args(
    default_timesteps=5000000, default_reward=1000000.0, default_iters=100000
)

# See the following links for benchmark results of other libraries:
#   Original paper: https://arxiv.org/abs/1812.05905
#   CleanRL: https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Mujoco--VmlldzoxODE0NjE
#   AgileRL: https://github.com/AgileRL/AgileRL?tab=readme-ov-file#benchmarks
benchmark_envs = {
    "HalfCheetah-v4": {
        "num_env_steps_sampled_lifetime": 1000000,
    },
    "Hopper-v4": {
        "num_env_steps_sampled_lifetime": 1000000,
    },
    "InvertedPendulum-v4": {
        "num_env_steps_sampled_lifetime": 1000000,
    },
    "InvertedDoublePendulum-v4": {
        "num_env_steps_sampled_lifetime": 1000000,
    },
    "Reacher-v4": {"num_env_steps_sampled_lifetime": 1000000},
    "Swimmer-v4": {"num_env_steps_sampled_lifetime": 1000000},
    "Walker2d-v4": {
        "num_env_steps_sampled_lifetime": 1000000,
    },
}


if __name__ == "__main__":
    args = parser.parse_args()

    metric = "evaluation_results/env_runner_results/episode_return_mean"
    mode = "max"
    num_env_runners = args.num_env_runners
    num_envs_per_env_runner = 2

    experiment_start_time = time.time()
    # Following the paper.
    for env, stop_criteria in benchmark_envs.items():
        hp_trial_start_time = time.time()
        config = (
            PPOConfig()
            .environment(env=env)
            # Enable new API stack and use EnvRunner.
            .api_stack(
                enable_env_runner_and_connector_v2=True,
                enable_rl_module_and_learner=True,
            )
            .env_runners(
                rollout_fragment_length="auto",
                num_env_runners=num_env_runners,
                num_envs_per_env_runner=num_envs_per_env_runner,
                env_to_module_connector=lambda env: MeanStdFilter(),
            )
            .resources(
                # Let's start with a small number of learner workers and
                # add later a tune grid search for these resources.
                # TODO (simon): Either add tune grid search here or make
                # an extra script to only test scalability.
                num_learner_workers=1,
                num_gpus_per_learner_worker=0.25,
            )
            .rl_module(
                model_config_dict={
                    "fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
                    "vf_share_layers": True,
                },
            )
            .training(
                lr=tune.grid_search([1e-5, 1e-3]),
                gamma=0.99,
                lambda_=0.95,
                entropy_coeff=tune.grid_search([0.0, 0.01]),
                vf_loss_coeff=tune.grid_search([0.1, 1.0]),
                clip_param=0.3,
                kl_target=0.02,
                mini_batch_size_per_learner=128,
                num_sgd_iter=32,
                vf_share_layers=True,
                use_kl_loss=tune.choice([False, True]),
                kl_coeff=0.2,
                vf_clip_param=float("inf"),
                grad_clip=float("inf"),
                train_batch_size_per_learner=4096,
            )
            .reporting(
                metrics_num_episodes_for_smoothing=5,
                min_sample_timesteps_per_iteration=1000,
            )
            .evaluation(
                evaluation_duration="auto",
                evaluation_duration_unit="timesteps",
                evaluation_interval=1,
                evaluation_num_env_runners=1,
                evaluation_parallel_to_training=True,
                evaluation_config={
                    # PPO learns stochastic policy.
                    "explore": False,
                },
            )
        )

        # TODO (sven, simon): The WandB callback has to be fixed by the
        # tune team. The current implementation leads to an OOM.
        callbacks = None
        if hasattr(args, "wandb_key") and args.wandb_key is not None:
            project = args.wandb_project or (
                "ppo-benchmarks-mujoco-grid-search"
                + "-"
                + re.sub("\\W+", "-", str(config.env).lower())
            )
            callbacks = [
                WandbLoggerCallback(
                    api_key=args.wandb_key,
                    project=project,
                    upload_checkpoints=True,
                    **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
                )
            ]

        tuner = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=train.RunConfig(
                stop={"num_env_steps_sampled_lifetime": args.stop_timesteps},
                storage_path="~/default/ray/bm_results",
                name="benchmark_ppo_mujoco_grid_search_" + env,
                callbacks=callbacks,
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=args.checkpoint_freq,
                    checkpoint_at_end=args.checkpoint_at_end,
                ),
            ),
            tune_config=tune.TuneConfig(
                num_samples=args.num_samples,
            ),
        )
        print(f"=========== Starting HP search for (env={env}) ===========")
        result_grid = tuner.fit()
        # Get the best result for the current environment.
        best_result = result_grid.get_best_result(metric=metric, mode=mode)
        print("-----------------------------------------------------------")
        print(
            f"Finished running HP search for (env={env}) in "
            f"{time.time() - hp_trial_start_time} seconds."
        )
        print(f"Best result for {env}: {best_result}")
        print(f"Best config for {env}: {best_result.config}")

        # Run again with the best config.
        best_trial_start_time = time.time()
        tuner = tune.Tuner(
            "PPO",
            param_space=best_result.config,
            run_config=train.RunConfig(
                stop=stop_criteria,
                name="benchmark_ppo_mujoco_grid_search_" + env + "_best",
            ),
        )
        print(f"=========== Best config run for (env={env}) ===========")
        print(f"Running best config for (env={env})...")
        tuner.fit()
        print(
            f"Finished running best config for (env={env}) "
            f"in {time.time() - best_trial_start_time} seconds."
        )
        print("--------------------------------------------------------")

    print(
        f"Finished running HP search on all MuJoCo benchmarks in "
        f"{time.time() - experiment_start_time} seconds."
    )
    print(
        "Results from running the best configs can be found in the "
        "`benchmark_ppo_mujoco_grid_search_<ENV-NAME>_best` directories."
    )
