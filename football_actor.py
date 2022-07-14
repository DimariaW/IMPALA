
import numpy as np
from collections import defaultdict

import gfootball.env as gfootball_env
from football_env import FootballEnv
from football_model import FootballNet, batchify
from football_agent import FootballAgent

from parl.env.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from parl.env.vector_env import VectorEnv
import paddle
import impala


#@parl.remote_class
class Actor(object):
    def __init__(self, config):
        paddle.device.set_device("cpu")
        self.config = config

        self.envs = []
        for _ in range(config['env_num']):
            env = gfootball_env.create_environment(env_name="11_vs_11_kaggle",
                                                   representation="raw",
                                                   rewards="scoring,checkpoints")
            env = FootballEnv(env)
            env = MonitorEnv(env)
            self.envs.append(env)
        self.vector_env = VectorEnv(self.envs)

        self.obs_batch = self.vector_env.reset()

        model = FootballNet()
        algorithm = impala.IMPALA(
            model,
            sample_batch_steps=self.config['sample_batch_steps'],
            gamma=self.config['gamma'],
            vf_loss_coeff=self.config['vf_loss_coeff'],
            clip_rho_threshold=self.config['clip_rho_threshold'],
            clip_pg_rho_threshold=self.config['clip_pg_rho_threshold'])
        self.agent = FootballAgent(algorithm)

    def sample(self):
        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)

        for i in range(self.config['sample_batch_steps']):
            actions, behaviour_logits = self.agent.sample(
                batchify(self.obs_batch, unsqueeze=0))
            #print(actions)
            #print(actions.shape)
            #print(behaviour_logits.shape)
            next_obs_batch, reward_batch, done_batch, info_batch = \
                    self.vector_env.step(actions)

            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(actions[env_id])
                env_sample_data[env_id]['behaviour_logits'].append(
                    behaviour_logits[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])

            self.obs_batch = next_obs_batch

        # Merge data of envs
        sample_data = defaultdict(list)
        for env_id in range(self.config['env_num']):
            for data_name in [
                    'obs', 'actions', 'behaviour_logits', 'rewards', 'dones'
            ]:
                sample_data[data_name].extend(
                    env_sample_data[env_id][data_name])

        # size of sample_data: env_num * sample_batch_steps
        for key in sample_data:
            sample_data[key] = batchify(sample_data[key], unsqueeze=0)

        return sample_data

    def get_metrics(self):
        metrics = defaultdict(list)
        for env in self.envs:
            monitor = get_wrapper_by_cls(env, MonitorEnv)
            if monitor is not None:
                for episode_rewards, episode_steps in monitor.next_episode_results(
                ):
                    metrics['episode_rewards'].append(episode_rewards)
                    metrics['episode_steps'].append(episode_steps)
        return metrics

    def set_weights(self, weights):
        self.agent.set_weights(weights)


if __name__ == "__main__":
    from impala_config import config
    actor = Actor(config)
    data = actor.sample()
    info = actor.agent.learn(data['obs'], data['actions'], data['behaviour_logits'], data['rewards'], data['dones'],
                             0.001, -0.01)
    assert(True)
