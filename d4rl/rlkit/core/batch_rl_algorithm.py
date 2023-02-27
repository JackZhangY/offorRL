import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            total_training_steps,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            num_epochs_per_log_interval=1,
            online_finetune=False # whether to implement offline2online
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.total_training_steps = total_training_steps
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.num_epochs_per_log_interval = num_epochs_per_log_interval
        self.online_finetune = online_finetune

        assert int((self.num_epochs - self._start_epoch) * self.num_train_loops_per_epoch * self.num_trains_per_train_loop)\
               == int(self.total_training_steps), 'mismatch of total training steps indicated in \'trainer\' and \'algorithm\''
        assert self._start_epoch < 0 and self.num_epochs >= 0, 'not satisfy epoch setting for offline RL'
        if self.online_finetune:
            assert self.num_epochs > 0, 'not satisfy epoch setting for offline2online RL'

    def _begin_epoch(self, epoch):
        # turn on the logging of learned variable at each log interval
        self.trainer.end_epoch(epoch)

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            if self.epoch == self._start_epoch or (self.epoch + 1) % self.num_epochs_per_log_interval == 0:
                self._begin_epoch(self.epoch)

            self._train()

            if self.epoch == self._start_epoch or (self.epoch + 1)% self.num_epochs_per_log_interval == 0:
                self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        if self.epoch == self._start_epoch or (self.epoch + 1)% self.num_epochs_per_log_interval == 0:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            if not self.offline_rl:
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)
