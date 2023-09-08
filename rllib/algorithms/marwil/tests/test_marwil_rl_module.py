import itertools
import unittest
import ray

from pathlib import Path
class TestMARWIL(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ray.init()

    @classmethod
    def tearDown(self) -> None:
        ray.shutdown()

    def test_rollouts(self):
        frameworks = ["tf2"]
        envs = ["CartPole-v1"]
        fwd_fns = ["forward_exploration", "forward_inference"]
        config_combinations = [frameworks, envs, fwd_fns]
        rllib_dir = Path(__file__).parents[3]
        print(f"rllib_dir={rllib_dir.as_posix()}")
        data_file = rllib_dir.joinpath("tests/data/cartpole/large.json")
        print(f"data_file={data_file.as_posix()}")

    
        
        for config in itertools.product(*config_combinations):
            fw, env, fwd_fn = config

            print(f"[Fw={fw}] | [Env={env}] | [FWD={fwd_fn}]")
            
            sample_batch


    