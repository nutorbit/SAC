import gym
import click
import torch
import numpy as np

from sac.sac import SAC


@click.command()
@click.argument('path')
def main(path):
    env = gym.make('CarRacing-v0')
    model = SAC(env.observation_space, env.action_space)

    actor, critic = torch.load(path)

    model.load_model(actor, critic)

    while True:
        obs = env.reset(random_position=False)
        done = False
        rews = []
        while not done:
            act = model.select_action(obs, evaluate=True)
            obs, rew, done, _ = env.step(act)
            rews.append(rew)
        print(np.sum(rews))


if __name__ == '__main__':
    main()
