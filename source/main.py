import gym
import numpy as np
from agents.keras_dense_agent.keras_agent import KerasAgent
from testtt import DQNAgent

env = gym.make('CartPole-v1')
agent = KerasAgent(state_size=env.observation_space.shape[0],
                   action_size=env.action_space.n)

for epi in range(5000):
    state = env.reset()
    state = np.reshape(state, [1,4])
    for t in range(5000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        reward = reward if not done else -10
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episode", epi, 'score', t, 'epsi:', agent.epsilon)
            break
    if len(agent.memory) > 32:
        agent.learn()