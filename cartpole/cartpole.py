import gym
import tensorflow as tf
import numpy as np
import matplotlib as plt
import random
from collections import deque

MEMORY_SIZE = 10000
BATCH_SIZE = 128
NUM_EPISODES = 50000
GAMMA = 0.9

class ReplayMemory():
    def __init__(self):
        self.memory = deque([], maxlen=MEMORY_SIZE)
        print("Created new replay memory")
    
    def add(self, item):
        self.memory.append(item)
        
    def sample_batch(self):
        sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*sample)
        batch = []
        for x in [states, actions, rewards, next_states, dones]:
            batch.append(tf.constant(x))
        return batch
        
    def __len__(self):
        return len(self.memory)
        
        
class DQN(tf.keras.Model):
    def __init__(self, observation_space_size, action_space_size):
        super().__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_space_size, activation=None)
        
    def call(self, x):
        hl1 = self.hidden_layer_1(x)
        hl2 = self.hidden_layer_2(hl1)
        out = self.output_layer(hl2)
        return out
        
  
class Cartpole():
    def __init__(self):
        env = gym.make("CartPole-v1")
        observation_space_size = env.observation_space.shape[0]      #4
        action_space_size = env.action_space.n                       #2
        self.dqn = DQN(observation_space_size, action_space_size)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        [states, actions, rewards, next_states, dones] = self.memory.sample_batch()
        
        #target_Q(s',a) for each next state s' in the batch and all actions a; [BS, num_actions]:
        next_state_predictions = self.target_dqn(next_states)
        
        #max_{a}(target_Q(s',a)) for each next state s' in the batch; [BS]:
        max_next_state_predictions = tf.reduce_max(next_state_predictions, axis=1)
        
        #mask that is 0 if the state is final, 1 otherwise
        final_state_mask = 1 - tf.cast(dones, tf.float32)
        
        #the training targets that Q(s,a) should match as closely as possible
        targets = rewards + GAMMA*(final_state_mask * max_next_state_predictions)
        
        with tf.GradientTape() as tape:
            #Q(s,a) for each state s in the batch and all actions a; [BS, num_actions]:
            all_predictions = self.dqn(states)
            
            #Q(s,a) for each state-action-pair (s,a) in the batch; [BS]:
            predictions = tf.gather(all_predictions, actions, axis=1, batch_dims=1)
            
            loss = self.loss_function(targets, predictions)
            
        gradients = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))
        
        return loss

    def training(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        env = gym.make("CartPole-v1")
        observation_space_size = env.observation_space.shape[0]      #4
        action_space_size = env.action_space.n                       #2
        print(f"Observation space size: {observation_space_size}, action space size: {action_space_size}")
        
        self.memory = ReplayMemory()
        self.target_dqn = DQN(observation_space_size, action_space_size)
        self.dqn.build((None, observation_space_size))
        self.target_dqn.build((None, observation_space_size))
        self.dqn.summary()
        self.target_dqn.summary()
        self.target_dqn.set_weights(self.dqn.get_weights())
        self.loss_function = tf.keras.losses.Huber()
        
        losses = []
        num_steps = []
        
        max_steps_managed = 0 #counts for how many episodes in a row the max number of steps was managed
        num_episodes_done = 0
        
        while num_episodes_done < NUM_EPISODES:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=max(0.001-0.0000001*num_episodes_done,0.0001))
            state = env.reset()
            done = False
            num_steps_managed = 0
            print(f"Episode {num_episodes_done+1}")
            while not done:
                exploration_rate = max(0., 0.5-0.0001*num_episodes_done)
                if random.random() < exploration_rate:
                    action = random.randrange(action_space_size)
                else:
                    prediction = self.dqn(tf.reshape(state, [1, observation_space_size]))
                    action = tf.argmax(prediction[0]).numpy()
                next_state, reward, done, info = env.step(action)
                self.memory.add((state, action, reward, next_state, done))
                state = next_state
                num_steps_managed += 1
            
            loss = self.train_step()
            print(f"Loss for episode {num_episodes_done}: {loss};\tmanaged {num_steps_managed} steps")
            
            num_episodes_done += 1
            if num_episodes_done % 10 == 0:
                self.target_dqn.set_weights(self.dqn.get_weights())
                print("Copied weights to target model")
            if num_episodes_done % 100 == 0:
                losses.append(loss.numpy())
                num_steps.append(num_steps_managed)
            if num_steps_managed == 500:
                max_steps_managed += 1
            else:
                max_steps_managed = 0
            if max_steps_managed == 25:
                self.dqn.save_weights('./checkpoints/my_checkpoint')
                print("Saved model")
                break
            
        env.close()
        print(losses)
        print(num_steps)
        
    def inference(self, num_games):
        ckpt = './checkpoints/my_checkpoint'
        self.dqn.load_weights(ckpt)
        print(f"Restored model from checkpoint {ckpt}")
        env = gym.make("CartPole-v1")
        observation_space_size = env.observation_space.shape[0]      #4
        action_space_size = env.action_space.n                       #2
        for game in range(num_games):
            state = env.reset()
            done = False
            num_steps_managed = 0
            while not done:
                env.render()
                prediction = self.dqn(tf.reshape(state, [1, observation_space_size]))
                action = tf.argmax(prediction[0]).numpy()
#                print(f"Prediction: {prediction[0].numpy()}")
#                print(f"Diff. prediction: {tf.reduce_max(prediction[0]).numpy() - tf.reduce_min(prediction[0]).numpy()}")
                next_state, _, done, _ = env.step(action)
                state = next_state
                num_steps_managed += 1
            print(f"Game {game}: managed {num_steps_managed} steps")
        env.close()
            
        
    
if __name__ == "__main__":
    cartpole = Cartpole()
    if input("Perform training? (y/n) ... ") == 'y':
        cartpole.training()
    num_games = int(input("Enter number of games to play ... "))
    cartpole.inference(num_games)
