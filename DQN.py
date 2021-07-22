import gym
import random
import tensorflow as tf
import numpy as np
import os
from collections import deque

output_dir = "model_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 32


class DQNAgent:

    def __init__(self, state_size, action_size, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(
            maxlen=2000)  # memory replay. randomly taking samples from all the episodes played for efficiency and diversity sake . episodes too close to each other have to be very similar and not of much use .
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # use linear since q is a regression problem

        ])
        Eve = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(metrics=["accuracy"], loss="mse", optimizer=Eve)
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            x = self.model.predict(state)
            return np.argmax(x)

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            if done:
                target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(new_state))  # Bellman function
            target_ = self.model.predict(state)
            print(target_, "a")
            target_[0][action] = target
            print(target_)
            self.model.fit(state, target_, epochs=10, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_episodes = 100
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    lr = 0.001
    agent = DQNAgent(state_size, action_size, gamma, epsilon, epsilon_decay, epsilon_min, lr)

    for i in range(n_episodes):
        done = False
        observation = env.reset()
        observation_shaped = np.reshape(observation, [1, state_size])
        while not done:
            action = agent.get_action(observation_shaped)
            observation_, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            observation_ = np.reshape(observation_, [1, state_size])
            agent.remember(observation_shaped, action, reward, observation_, done)
            observation_shaped = observation_

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if i % 50 == 0:
            agent.save(output_dir + "   weights_" + '{:04d}'.format(i) + ".h5")

    for i in range(100):
        done = False
        qqqq = env.reset()
        qqqq = np.reshape(qqqq, [1, state_size])
        while not done:
            action = agent.get_action(qqqq)
            env.render(action)
            qqqq, _, done, _ = env.step(action)
            qqqq = np.reshape(qqqq, [1, state_size])
