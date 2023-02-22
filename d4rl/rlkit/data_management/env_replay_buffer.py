from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import h5py
from tqdm import tqdm


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            online_finetune=False,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.env_name = self.env.spec.name
        self.online_finetune = online_finetune

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


    def load_hdf5(self, buffer_filename, terminate_on_end=False):

        data_dict = {}
        with h5py.File(buffer_filename, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        assert self._max_replay_buffer_size >= N_samples, "dataaset does not fit in replay buffer"
        if self._ob_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self._ob_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self._ob_space.shape))
        assert data_dict['actions'].shape[1:] == self._action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self._action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['terminals'].shape))

        # Returns datasets formatted for use by standard Q-learning algorithms,
        # with observations, actions, next_observations, rewards, and a terminalflag.

        N = data_dict['rewards'].shape[0]
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in data_dict:
            use_timeouts = True

        episode_step = 0
        for i in range(N-1):
            obs = data_dict['observations'][i].astype(np.float32)
            new_obs = data_dict['observations'][i+1].astype(np.float32)
            action = data_dict['actions'][i].astype(np.float32)
            reward = data_dict['rewards'][i].astype(np.float32)
            done_bool = bool(data_dict['terminals'][i])

            if use_timeouts:
                final_timestep = data_dict['timeouts'][i]
            else:
                final_timestep = (episode_step == self.env._max_episode_steps - 1)
            if (not terminate_on_end) and final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1

        # add to current replaybuffer
        if not self.online_finetune:
            self._observations = np.array(obs_)
            self._next_obs = np.array(next_obs_)
            self._actions = np.array(action_)

            if 'antmaze' in self.env_name:
                self._rewards = (np.expand_dims(np.squeeze(np.array(reward_)), 1) - 0.5) * 4.0
            else:
                self._rewards = np.expand_dims(np.squeeze(np.array(reward_)), 1)

            self._terminals = np.expand_dims(np.squeeze(np.array(done_)), 1)
            self._size = np.array(done_).shape[0]
            print ('Number of terminals on: ', self._terminals.sum())
            self._top = self._size
            self._offline_size = self._size
            print('Total samples number: {}'.format(self._size))

        else:
            self._size = np.array(done_).shape[0]
            print ('Number of terminals on: ', self._terminals.sum())
            self._top = self._size
            self._offline_size = self._size
            print('Total samples number: {}'.format(self._size))

            self._observations[:self._size] = np.array(obs_)
            self._next_obs[:self._size] = np.array(next_obs_)
            self._actions[:self._size] = np.array(action_)

            if 'antmaze' in self.env_name:
                self._rewards[:self._size] = (np.expand_dims(np.squeeze(np.array(reward_)), 1) - 0.5) * 4.0
            else:
                self._rewards[:self._size] = np.expand_dims(np.squeeze(np.array(reward_)), 1)

            self._terminals[:self._size] = np.expand_dims(np.squeeze(np.array(done_)), 1)


    def normalize_states(self, eps=1e-3):
        self.mean = self._observations.mean(axis=0, keepdims=True)
        self.std = self._observations.std(axis=0, keepdims=True) + eps
        self._observations = (self._observations - self.mean) / self.std
        self._next_obs = (self._next_obs - self.mean) / self.std
        print('=============  finishing state normalization  =================')

    def reward_scale_by_traj_returns(self):
        assert self._offline_size > 0, 'please load offline dataset for reward scale'
        returns, lengths = [], []
        ep_ret, ep_len = 0., 0
        for r, d in zip(self._rewards[:self._offline_size].squeeze(), self._terminals[:self._offline_size].squeeze()):
            ep_ret += float(r)
            ep_len += 1
            if d or ep_len == self.env._max_episode_steps:
                returns.append(ep_ret)
                lengths.append(ep_len)
                ep_ret, ep_len = 0., 0
        lengths.append(ep_len)
        assert sum(lengths) == self._offline_size, 'miscount number of offline data'
        reward_scale = self.env._max_episode_steps / (max(returns) - min(returns) + 1E-8)
        return reward_scale

