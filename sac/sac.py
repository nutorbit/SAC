import torch

from torch.optim import Adam
from cpprb import ReplayBuffer

from sac.model import ActorCNN, CriticCNN
from sac.utils import huber_loss, get_default_rb_dict, Logger


class SAC:
    """
    Soft Actor Critic
    Ref: https://arxiv.org/pdf/1812.05905.pdf
    """

    def __init__(self, observation_space, action_space,
                 replay_size=int(1e6),
                 gamma=0.99,
                 tau=0.05,
                 lr=3e-4,
                 alpha=0.2,
                 target_update_interval=1,
                 device='cuda'):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.device = device
        self.logger = Logger()

        # Experience replay
        rb_kwargs = get_default_rb_dict(observation_space.shape, action_space.shape, replay_size)
        self.rb = ReplayBuffer(**rb_kwargs)

        # critic
        self.critic = CriticCNN(obs_dim=observation_space.shape[0], act_dim=action_space.shape[0]).to(self.device)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # critic target
        self.critic_target = CriticCNN(obs_dim=observation_space.shape[0], act_dim=action_space.shape[0]).to(self.device)
        self.critic_target.hard_update(self.critic)

        # actor
        self.actor = ActorCNN(obs_dim=observation_space.shape[0], act_dim=action_space.shape[0],
                           action_space=action_space).to(self.device)
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=lr)

    def select_action(self, obs, evaluate=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(obs)
        else:
            _, _, action = self.actor.sample(obs)
        return action.detach().cpu().numpy()[0]

    def compute_td_error(self, obs, act, next_obs, rew, done):
        with torch.no_grad():
            next_act, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rew + ((1 - done) * self.gamma * target_q)

        current_q1, current_q2 = self.critic(obs, act)

        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q

        return td_error1, td_error2

    def critic_loss(self, obs, act, next_obs, rew, done):
        td_error1, td_error2 = self.compute_td_error(obs, act, next_obs, rew, done)

        # MSE
        loss1 = huber_loss(td_error1).mean()
        loss2 = huber_loss(td_error2).mean()

        return loss1 + loss2

    def actor_alpha_loss(self, obs):

        act, log_prob, _ = self.actor.sample(obs)

        current_q1, current_q2 = self.critic(obs, act)
        min_q = torch.min(current_q1, current_q2)

        actor_loss = ((self.alpha * log_prob) - min_q).mean()

        # alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss

    def update_critic(self, obs, act, next_obs, rew, done):
        loss = self.critic_loss(obs, act, next_obs, rew, done)

        # update q1
        self.critic_opt.zero_grad()
        loss.backward(retain_graph=True)
        self.critic_opt.step()

        return loss

    def update_actor_alpha(self, obs):
        actor_loss, alpha_loss = self.actor_alpha_loss(obs)

        # update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # update alpha
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return actor_loss, alpha_loss

    def update_parameters(self, batch_size, updates):

        batch = self.rb.sample(batch_size)

        # to tensor
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).to(self.device)
        done = torch.FloatTensor(batch['done']).to(self.device)

        # update actor & critic & alpha
        critic_loss = self.update_critic(obs, act, next_obs, rew, done)
        actor_loss, alpha_loss = self.update_actor_alpha(obs)

        # apply alpha
        self.alpha = self.log_alpha.exp()

        # update target network
        if updates % self.target_update_interval == 0:
            self.critic_target.soft_update(self.critic, self.tau)

        return critic_loss, actor_loss, alpha_loss, self.alpha.clone()

    def load_model(self, actor, critic):
        self.actor = actor
        self.critic = critic