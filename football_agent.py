
from football_model import to_tensor
import numpy as np
import parl
import paddle


class FootballAgent(parl.Agent):
    def __init__(self, algorithm):
        super().__init__(algorithm)

    def sample(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
            Format of image input should be NCHW format.

        Returns:
            predict_actions: a numpy int64 array of shape [B]
            behaviour_logits: a numpy float32 array of shape [B, act_dim]
        """
        obs = to_tensor(obs_np, unsqueeze=None)
        probs, behaviour_logits = self.alg.sample(obs)
        probs = probs.cpu().numpy()
        sample_actions = np.array(
            [np.random.choice(len(prob), 1, p=prob)[0] for prob in probs])

        return sample_actions, behaviour_logits.cpu().numpy()

    def learn(self, obs_np, actions_np, behaviour_logits_np, rewards_np,
              dones_np, lr, entropy_coeff):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.
            actions_np: a numpy int64 array of shape [B]
            behaviour_logits_np: a numpy float32 array of shape [B, act_dim]
            rewards_np: a numpy float32 array of shape [B]
            dones_np: a numpy bool array of shape [B]
            lr: float scalar of learning rate.
            entropy_coeff: float scalar of entropy coefficient.
        """

        obs = to_tensor(obs_np)
        actions = to_tensor(actions_np)
        behaviour_logits = to_tensor(
            behaviour_logits_np)
        rewards = to_tensor(rewards_np)
        dones = to_tensor(dones_np)

        vtrace_loss, kl = self.alg.learn(obs, actions, behaviour_logits,
                                         rewards, dones, lr, entropy_coeff)

        total_loss = vtrace_loss.total_loss.cpu().numpy()
        pi_loss = vtrace_loss.pi_loss.cpu().numpy()
        vf_loss = vtrace_loss.vf_loss.cpu().numpy()
        entropy = vtrace_loss.entropy.cpu().numpy()
        kl = kl.cpu().numpy()

        return total_loss, pi_loss, vf_loss, entropy, kl

    def predict(self):
        pass
