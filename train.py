import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from env.envs import LazyVicsekEnv, Config, load_config
from model.model_rllib import LazyListenerModelPPOTest

if __name__ == "__main__":

    do_debug = False
    # do_debug = True

    if do_debug:
        ray.init(local_mode=True)

    # Set up environment configuration
    default_config_path = "D:/pych_ws/lazy-vicsek/env_config_train.yaml"
    my_config = load_config(default_config_path)
    my_config.env.num_agents_pool = [20]
    my_config.env.max_time_steps = 100
    my_config.env.alignment_goal = 0.96
    my_config.env.alignment_rate_goal = 0.04
    my_config.env.use_fixed_episode_length = True
    my_config.control.speed = 15
    my_config.control.max_turn_rate = 1e3  # classical Vicsek models adopt particle dynamics

    # register your custom environment
    env_config = {
        "seed_id": None,
        "config": my_config.dict(),  # pass dict to save the config
    }
    env_name = "vicsek_lazy_env"
    register_env(env_name, lambda cfg: LazyVicsekEnv(cfg))

    # register your custom model
    model_name = "vicsek_lazy_listener"
    ModelCatalog.register_custom_model(model_name, LazyListenerModelPPOTest)

    # train
    tune.run(
        "PPO",
        name="lazy_initial_test_240618",
        # resume=True,
        # stop={"episode_reward_mean": -101},
        # stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=12,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            #
            # "callbacks": MyCallbacks,
            #
            "model": {
                "custom_model": model_name,
                # "custom_model_config": custom_model_config,
                # "custom_action_dist": "det_cont_action_dist" if custom_model_config["use_deterministic_action_dist"] else None,
            },
            "num_gpus": 0.5,
            "num_workers": 14,
            "num_envs_per_worker": 3,
            "rollout_fragment_length": 500,
            "train_batch_size": 21000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            # "batch_mode": "complete_episodes",
            # "batch_mode": "truncate_episodes",
            "lr": 2e-5,
            # "lr_schedule": [[0, 2e-5],
            #                 [1e7, 1e-7],
            #                 ],
            # add more hyperparameters here as needed
            #############################
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": 0.25,
            # In the...
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0,  # no PPO penalty term; we use PPO-clip anyway; if none zero, be careful Nan in tensors!
            # "entropy_coeff": tune.grid_search([0, 0.001, 0.0025, 0.01]),
            # "entropy_coeff_schedule": None,
            # "entropy_coeff_schedule": [[0, 0.003],
            #                            [5e4, 0.002],
            #                            [1e5, 0.001],
            #                            [2e5, 0.0005],
            #                            [5e5, 0.0002],
            #                            [1e6, 0.0001],
            #                            [2e6, 0],
            #                            ],
            "clip_param": 0.2,  # 0.3
            "vf_clip_param": 256,
            # "grad_clip": None,
            "grad_clip": 0.5,
            "kl_target": 0.01,
        },
    )