import gym
import numpy as np

from tqdm import trange

from sac.sac import SAC


REPLAY_SIZE = int(1e3)
N_STEPS = int(1e6)
START_STEPS = int(1e4)
BATCH_SIZE = 128
STEPS_PER_EPOCHS = 4000
RENDER = False


def evaluate(env, agent):
    rews, steps = [], []
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    while not done:
        act = agent.select_action(obs, evaluate=True)
        # print(act)
        obs, rew, done, _ = env.step(act)
        episode_reward += rew
        episode_steps += 1
    steps.append(episode_steps)
    rews.append(episode_reward)

    return np.mean(rews), np.mean(steps)


def main():
    env = gym.make('CarRacing-v0')
    model = SAC(env.observation_space, env.action_space, replay_size=REPLAY_SIZE, device='cuda')
    logger = model.logger

    updates = 0
    best_to_save = float('-inf')
    episode_rew, episode_steps = 0, 0
    obs = env.reset()

    for t in trange(N_STEPS):
        if RENDER:
            env.render()

        if t < START_STEPS:
            act = env.action_space.sample()
        else:
            act = model.select_action(obs)

        next_obs, rew, done, _ = env.step(act)

        episode_rew += rew
        episode_steps += 1

        model.rb.add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)

        obs = next_obs

        # reset when terminated
        if done:
            n_collision = env.report()

            obs = env.reset(random_position=False)

            logger.store('Reward/train', episode_rew)
            logger.store('Steps/train', episode_steps)
            logger.store('N_Collision/train', n_collision)

            episode_rew, episode_steps = 0, 0

        # update nn
        if model.rb.get_stored_size() > BATCH_SIZE:
            critic_loss, actor_loss, alpha_loss, alpha = model.update_parameters(BATCH_SIZE, updates)

            logger.store('Loss/Critic', critic_loss)
            logger.store('Loss/Actor', actor_loss)
            logger.store('Loss/Alpha', alpha_loss)
            logger.store('Param/Alpha', alpha)

            updates += 1

        # eval and save
        if (t + 1) % STEPS_PER_EPOCHS == 0:
            n_collision = env.report()

            # test
            mean_rew, mean_steps = evaluate(env, model)
            logger.store('Reward/test', mean_rew)
            logger.store('Steps/test', mean_steps)
            logger.store('N_Collision/test', n_collision)

            # save a model
            if best_to_save <= mean_rew:
                best_to_save = mean_rew
                logger.save_model([model.actor, model.critic])

        logger.update_steps()


if __name__ == "__main__":
    main()

