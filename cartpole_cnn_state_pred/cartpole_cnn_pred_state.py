import gym
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import logging
from collections import deque

import os

MEMORY_SIZE = 10000
BATCH_SIZE = 128
NUM_EPISODES = 100000
GAMMA = 0.9

class ReplayMemory():
    def __init__(self):
        self.memory = deque([], maxlen=MEMORY_SIZE)
        print("Created new replay memory")
    
    def add(self, item):
        self.memory.append(item)
        
    def sample_batch(self):
        sample = random.sample(self.memory, BATCH_SIZE)
        
        screens, states, actions, rewards, next_screens, next_states, dones = zip(*sample)
        batch = []
        for x in [screens, states, actions, rewards, next_screens, next_states, dones]:
            batch.append(tf.constant(x))
        return batch
        
    def __len__(self):
        return len(self.memory)
        
  
class DQN(tf.keras.Model):
    def __init__(self, action_space_size):
        super().__init__()                                                              #[BS,160,240,3]
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(64, 5, strides=(3, 3), activation='relu')  #-> [BS,52,79,64]
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(64, 5, strides=(2, 2), activation='relu')  #-> [BS,25,38,64]
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides=(1, 1), activation='relu')  #-> [BS,23,36,64]
        self.flatten = tf.keras.layers.Flatten()                                        #-> [BS,52992]
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.fc_0 = tf.keras.layers.Dense(512, activation='relu')                       #-> [BS,512]
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.fc_1 = tf.keras.layers.Dense(256, activation='relu')                       #-> [BS,256]
        self.bn_6 = tf.keras.layers.BatchNormalization()
        self.fc_3 = tf.keras.layers.Dense(64, activation='relu')                        #-> [BS,64]
        self.pred_state = tf.keras.layers.Dense(4, activation= None)                    #-> [BS,4]
        self.fc_4 = tf.keras.layers.Dense(128, activation='relu')                       #-> [BS,128]
        self.fc_5 = tf.keras.layers.Dense(128, activation='relu')                       #-> [BS,128]
        self.output_layer = tf.keras.layers.Dense(action_space_size, activation=None)   #-> [BS,2]

    def call(self, x):
        x = self.bn_1(x)
        x = self.conv_1(x)
        x = self.bn_2(x)
        x = self.conv_2(x)
        x = self.bn_3(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.bn_4(x)
        x = self.fc_0(x)
        x = self.bn_5(x)
        x = self.fc_1(x)
        x = self.bn_6(x)
        x = self.fc_3(x)
        pred_state = self.pred_state(x)
        # the following layers are not used, the reward prediction is handled by the state_DQN
        x = self.fc_4(pred_state)
        x = self.fc_5(x)
        out = self.output_layer(x)
        return pred_state, out
        
        
class state_DQN(tf.keras.Model):
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
        
        self.dqn = DQN(action_space_size)
        
        self.state_dqn = state_DQN(observation_space_size,action_space_size)
        self.state_dqn.build((None, observation_space_size))
        
        state_dqn_ckpt = '../cartpole/checkpoints/my_checkpoint'
        self.state_dqn.load_weights(state_dqn_ckpt)
        print(f"restored state dqn from {state_dqn_ckpt}")

    def pre_train_step(self):
        # trains the first part of the model to infer the state from the input screens
        if len(self.memory) < BATCH_SIZE:
            return
        [screens, states, actions, rewards, next_screens, next_states, dones] = self.memory.sample_batch()
        mse = tf.keras.losses.MeanSquaredError()
        
        with tf.GradientTape() as tape:
            pred_states, _ = self.dqn(screens)
            
            #Empirically over 1000 episodes, mean(abs(state)) = [0.04971903 0.59890999 0.06660783 0.92719498]; To account for these different magnitudes, the following scaling is applied:
            scaling = [10., 1., 10., 1.]
            #This is not the same as normalizing by subtracting the mean (which is zero here) and dividing by the standard deviation, but for a rough loss scaling, this should be sufficient
            loss = mse(scaling*pred_states, scaling*states)
            
        gradients = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))
        
        print(f"predicted state = {pred_states[0]}\nGT state= {states[0]}")
        
        return loss

#    # Not used here
#    def train_step(self):
#        # trains the model end-to-end
#        if len(self.memory) < BATCH_SIZE:
#            return
#        [screens, states, actions, rewards, next_screens, next_states, dones] = self.memory.sample_batch()
#
#        #target_Q(s',a) for each next state s' in the batch and all actions a; [BS, num_actions]:
#        _, next_state_predictions = self.target_dqn(next_screens)
#
#        #max_{a}(target_Q(s',a)) for each next state s' in the batch; [BS]:
#        max_next_state_predictions = tf.reduce_max(next_state_predictions, axis=1)
#
#        #mask that is 0 if the state is final, 1 otherwise
#        final_state_mask = 1 - tf.cast(dones, tf.float32)
#
#        #the training targets that Q(s,a) should match as closely as possible
#        targets = rewards + GAMMA*(final_state_mask * max_next_state_predictions)
#
#        with tf.GradientTape() as tape:
#            #Q(s,a) for each state s in the batch and all actions a; [BS, num_actions]:
#            _, all_predictions = self.dqn(screens)
#
#            #Q(s,a) for each state-action-pair (s,a) in the batch; [BS]:
#            predictions = tf.gather(all_predictions, actions, axis=1, batch_dims=1)
#
#            loss = self.loss_function(targets, predictions)
#
#        gradients = tape.gradient(loss, self.dqn.trainable_variables)
#        self.optimizer.apply_gradients(zip(gradients, self.dqn.trainable_variables))
#
#        return loss

    def training(self):
        logging.basicConfig(filename='train_log.log', level=logging.INFO)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        env = gym.make("CartPole-v1")
        observation_space_size = env.observation_space.shape[0]      #4
        action_space_size = env.action_space.n                       #2
        screen_size = [160,240,3]
        print(f"Observation space size: {observation_space_size}, action space size: {action_space_size}")
        
        self.memory = ReplayMemory()
        self.target_dqn = DQN(action_space_size)
        self.dqn.build((None, *screen_size))
        self.target_dqn.build((None, *screen_size))
        self.dqn.summary()
        self.target_dqn.summary()
        self.target_dqn.set_weights(self.dqn.get_weights())
        self.loss_function = tf.keras.losses.Huber()
                
        losses = []
        num_steps = []
        
        max_steps_managed = 0 #counts for how many episodes in a row the max number of steps was managed
        num_episodes_done = 0
        
#        #Just to continue training from existing checkpoint
#        num_episodes_done = 600
#        self.dqn.load_weights('./checkpoints_pred_state_episode_13800/my_checkpoint')
#        print("restored model")
#        self.target_dqn.load_weights('./checkpoints_pred_state_episode_13800/my_checkpoint')
#        ##########
        
        while num_episodes_done < NUM_EPISODES:
            lr = (1e-3 / (1.+1e-3*num_episodes_done)) + 1e-5
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            state = env.reset()
            done = False
            num_steps_managed = 0
            print(f"Episode {num_episodes_done}")
            
            curr_screen = env.render(mode='rgb_array')
            curr_screen = cv2.cvtColor(curr_screen, cv2.COLOR_RGB2GRAY)
            curr_screen = cv2.resize(curr_screen, (screen_size[1], screen_size[0]), interpolation=cv2.INTER_CUBIC)
            curr_screen[curr_screen < 255] = 0
            curr_screen = curr_screen / 255
                        
            prev2_prev_curr_next_screen = np.stack((curr_screen, curr_screen, curr_screen, curr_screen), axis=-1)
            while not done:
                #env.render()
                exploration_rate = max(0.05, 0.5-0.0001*num_episodes_done)
                if random.random() < exploration_rate and num_episodes_done%10 != 9:
                    action = random.randrange(action_space_size)
                else:
                    pred_state, _ = self.dqn(tf.reshape(prev2_prev_curr_next_screen[:, :, :3], [1, *screen_size]))
                    prediction = self.state_dqn(pred_state)
                    action = tf.argmax(prediction[0]).numpy()
                
                next_state, reward, done, info = env.step(action)
                
                next_screen = env.render(mode='rgb_array')
                next_screen = cv2.cvtColor(next_screen, cv2.COLOR_RGB2GRAY)
                next_screen = cv2.resize(next_screen, (screen_size[1], screen_size[0]), interpolation=cv2.INTER_CUBIC)
                next_screen[next_screen < 255] = 0
                next_screen = next_screen / 255
                
                prev2_prev_curr_next_screen[:, :, 3] = next_screen
                
#                _, axarr = plt.subplots(4)
#                axarr[0].imshow(prev2_prev_curr_next_screen[:, :, 0], cmap='gray')
#                axarr[1].imshow(prev2_prev_curr_next_screen[:, :, 1], cmap='gray')
#                axarr[2].imshow(prev2_prev_curr_next_screen[:, :, 2], cmap='gray')
#                axarr[3].imshow(prev2_prev_curr_next_screen[:, :, 3], cmap='gray')
#                plt.show()
#                input("...")
                
                self.memory.add((prev2_prev_curr_next_screen[:, :, :3], state, action, reward, prev2_prev_curr_next_screen[:, :, 1:], next_state, done))
                
                state = next_state
                prev2_prev_curr_next_screen = np.concatenate((prev2_prev_curr_next_screen[:, :, 1:], np.expand_dims(next_screen, axis=-1)), axis=-1)
                num_steps_managed += 1
            
            loss = self.pre_train_step()
            print(f"Loss for episode {num_episodes_done}: {loss};\tmanaged {num_steps_managed} steps")
            print(f"Exploration rate = {exploration_rate if num_episodes_done%10 != 9 else 0.}, learning rate = {lr}")
            logging.info(f"Loss for episode {num_episodes_done}: {loss};\tmanaged {num_steps_managed} steps")
            logging.info(f"Exploration rate = {exploration_rate if num_episodes_done%10 != 9 else 0.}, learning rate = {lr}")
            
            num_episodes_done += 1
            if num_episodes_done % 10 == 0:
                self.target_dqn.set_weights(self.dqn.get_weights())
                print("Copied weights to target model")
            if num_episodes_done % 100 == 0 and not loss is None:
                losses.append(loss.numpy())
                num_steps.append(num_steps_managed)
            if num_steps_managed == 500:
                max_steps_managed += 1
            else:
                max_steps_managed = 0
            if num_episodes_done % 100 == 0:
                self.dqn.save_weights('./checkpoints/my_checkpoint')
                print("Saved model")
            if max_steps_managed == 25:
                self.dqn.save_weights('./checkpoints/my_checkpoint')
                print("Saved model")
                break
            
        env.close()
        print(losses)
        print(num_steps)
        self.dqn.save_weights('./checkpoints/my_checkpoint')
        print("Saved model")
        
    def inference(self, num_games):
        ckpt = './checkpoints_pred_state_episode_13800/my_checkpoint'
        self.dqn.load_weights(ckpt)
        print(f"Restored from checkpoint {ckpt}")
        env = gym.make("CartPole-v1")
        observation_space_size = env.observation_space.shape[0]      #4
        action_space_size = env.action_space.n                       #2
        screen_size = [160,240,3]
        
        for game in range(num_games):
            state = env.reset()
            done = False
            num_steps_managed = 0
            
            curr_screen = env.render(mode='rgb_array')
            curr_screen = cv2.cvtColor(curr_screen, cv2.COLOR_RGB2GRAY)
            curr_screen = cv2.resize(curr_screen, (screen_size[1], screen_size[0]), interpolation=cv2.INTER_CUBIC)
            curr_screen[curr_screen < 255] = 0
            curr_screen = curr_screen / 255
                        
            prev2_prev_curr_next_screen = np.stack((curr_screen, curr_screen, curr_screen, curr_screen), axis=-1)
            while not done:
                env.render()
                
                pred_state, _ = self.dqn(tf.reshape(prev2_prev_curr_next_screen[:, :, :3], [1, *screen_size]))
                prediction = self.state_dqn(pred_state)
                action = tf.argmax(prediction[0]).numpy()
                
                next_state, _, done, _ = env.step(action)
                
                next_screen = env.render(mode='rgb_array')
                next_screen = cv2.cvtColor(next_screen, cv2.COLOR_RGB2GRAY)
                next_screen = cv2.resize(next_screen, (screen_size[1], screen_size[0]), interpolation=cv2.INTER_CUBIC)
                next_screen[next_screen < 255] = 0
                next_screen = next_screen / 255
                prev2_prev_curr_next_screen[:, :, 3] = next_screen
                
                state = next_state
                prev2_prev_curr_next_screen = np.concatenate((prev2_prev_curr_next_screen[:, :, 1:], np.expand_dims(next_screen, axis=-1)), axis=-1)
                
                num_steps_managed += 1
            print(f"Game {game}: managed {num_steps_managed} steps")
        env.close()
            
        
    
if __name__ == "__main__":
    cartpole = Cartpole()
    if input("Perform training? (y/n) ... ") == 'y':
        os.environ["SDL_VIDEODRIVER"] = "dummy" #no game window is opened
        cartpole.training()
    num_games = int(input("Enter number of games to play ... "))
    cartpole.inference(num_games)
