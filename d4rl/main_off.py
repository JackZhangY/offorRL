import os
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import random
import numpy as np
import d4rl, gym
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import *
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.exploration_strategies import OUStrategy, GaussianAndEpsilonStrategy, PolicyWrappedWithExplorationStrategy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.sac import *
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.launcher_util import setup_logger, omegaconf_to_dict
from rlkit.core import logger

PolicyPool = {
    'GaussianPolicy': GaussianPolicy,
    'TanhGaussianPolicy': TanhGaussianPolicy,
    'VaePolicy': VaePolicy,
}
TrainerPool = {
    'IQL': IQLTrainer,
    'CQL': CQLTrainer,
    'S4RL': S4RLTrainer,
    'SQL': SQLTrainer,
    'BC': BCTrainer,
    'IKL': IKLTrainer,
}

def save_model(save_path, trainer):
    assert os.path.exists(save_path), 'no such config folder, should make this path before calling this...'
    torch.save(trainer.policy.state_dict(), os.path.join(save_path, 'final_policy.pth'))

@hydra.main(config_path='configs', config_name='base.yaml')
def main(args):

    torch.set_num_threads(2)

    if args.spec is not None:
        assert args.spec == args.env.name, 'env mismatches with the specific config'

    exp_prefix = args.trainer.exp_prefix
    base_log_dir = os.path.join(args.logger.log_dir, args.env.name, args.trainer.name)
    log_dir, variant_log_path = setup_logger(exp_prefix=exp_prefix, variant=omegaconf_to_dict(args),
                                             seed=args.device.seed, base_log_dir=base_log_dir,
                                             include_exp_prefix_sub_dir=False)

    ### device setting
    if args.device.cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device.gpu_idx)
        ptu.set_gpu_mode(True, args.device.gpu_idx)
    seed = args.device.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### env & buffer setting
    # env used for evaluation
    eval_env = make(args.env.name, None, None, normalize_env=args.trainer.obs_norm)
    if args.env.max_episode_steps > 0:
        eval_env._max_episode_steps = args.env.max_episode_steps
    eval_env.seed(seed)
    eval_env.action_space.seed(seed)

    # env used for online finetune
    if args.rlalg.online_finetune:
        expl_env = make(args.env.name, None, None, normalize_env=args.trainer.obs_norm)
        expl_env.seed(seed)
        expl_env.action_space.seed(seed)
        if args.env.max_episode_steps > 0:
            expl_env._max_episode_steps = args.env.max_episode_steps
    else:
        expl_env = None

    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    max_action = float(eval_env.action_space.high[0])
    if args.trainer.policy_kwargs.get('max_action') is not None:
        args.trainer.policy_kwargs.max_action = max_action

    # replay buffer
    replay_buffer = EnvReplayBuffer(args.buffer.max_replay_buffer_size, eval_env,
                                    online_finetune=args.rlalg.online_finetune)
    replay_buffer.load_hdf5()
    if args.trainer.obs_norm:
        replay_buffer.normalize_states()
        eval_env.set_obs_stats(replay_buffer.mean, replay_buffer.std)
        if args.rlalg.online_finetune:
            expl_env.set_obs_stats(replay_buffer.mean, replay_buffer.std)
        logger.log("finish the normalization setting in replay buffer and environment")
    if args.trainer.reward_norm:
        reward_scale = replay_buffer.reward_scale_by_traj_returns()
        args.trainer.trainer_kwargs.reward_scale = reward_scale

    ### AC network
    # critic network
    qf_kwargs = args.trainer.qf_kwargs
    if qf_kwargs is not None:
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
        target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
    else:
        qf1, qf2, target_qf1, target_qf2 = None, None, None, None

    if args.trainer.vf_kwargs is not None:
        vf = ConcatMlp(
            input_size=obs_dim,
            output_size=1,
            **args.trainer.vf_kwargs
        )
    else:
        vf = None

    # actor network
    policy = PolicyPool[args.trainer.policy_type](
        obs_dim=obs_dim,
        action_dim=action_dim,
        **args.trainer.policy_kwargs
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)

    if args.rlalg.online_finetune:
        assert expl_env is not None, 'online finetune needs an exploration env'
        expl_policy = policy # common gaussian stochastic policy for exploration

        online_expl_kwargs =  args.trainer.expl_kwargs
        if online_expl_kwargs is not None:
            expl_type = online_expl_kwargs['expl_type']
            if expl_type == 'ou':
                es = OUStrategy(
                    action_space=expl_env.action_space,
                    max_sigma=online_expl_kwargs['noise'],
                    min_sigma=online_expl_kwargs['noise'],
                )
                expl_policy = PolicyWrappedWithExplorationStrategy(
                    exploration_strategy=es,
                    policy=expl_policy,
                )

            elif expl_type == 'gauss_eps':

                es = GaussianAndEpsilonStrategy(
                    action_space=expl_env.action_space,
                    max_sigma=online_expl_kwargs['noise'],
                    min_sigma=online_expl_kwargs['noise'],  # constant sigma
                    epsilon=0,
                )
                expl_policy = PolicyWrappedWithExplorationStrategy(
                    exploration_strategy=es,
                    policy=expl_policy,
                )

        expl_path_collector = MdpPathCollector(expl_env, expl_policy)
    else:
        expl_path_collector = None


    ### init trainer
    trainer = TrainerPool[args.trainer.name](
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        **args.trainer.trainer_kwargs
    )

    ### init offline algorithm
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=eval_env._max_episode_steps,
        total_training_steps=args.trainer.trainer_kwargs.total_training_steps,
        **args.rlalg
    )

    # rewrite the 'variant.json' for some variant changes, e.g., 'reward_scale'
    logger.log_variant(variant_log_path, omegaconf_to_dict(args))

    algorithm.to(ptu.device)
    algorithm.train()

    save_model(log_dir, trainer)



if __name__ == '__main__':
    main()







