from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np  
import math 
from random import shuffle
import tensorflow.keras.backend as K
import tensorflow as tf
from copy import deepcopy

import os
# os.environ['CUDA_VISIBLE_DEVICES']=''

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# print(f'COMPLEX_MOVEMENT: {len(COMPLEX_MOVEMENT)}')

# # actions for more complex movement
# COMPLEX_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]

'''
 state.shape: (240, 256, 3), 
 reward: 0.0, 
 info: {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40, 'y_pos': 79}
'''

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
  
    # env.render()
   

env.close()










# agent for frequently updating
online_agent = Agent('online')

# agent for slow updating
target_agent = Agent('target')
target_agent.model.set_weights(online_agent.model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
average_loss = tf.keras.metrics.Mean(name='loss')


@tf.function
def train_step(state, action, reward, next_state, ternimal):
    
    tar_Q = target_agent.max_Q(next_state)

    with tf.GradientTape() as tape:
        loss = online_agent.loss(state, action, reward, tar_Q, ternimal)

    gradients = tape.gradient(loss, online_agent.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, online_agent.model.trainable_variables))

    average_loss.update_state(loss)



    
class Replay_buffer():

    def __init__(self, buffer_size=50000):
        self.experiences = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, size):
        """
        sample experience from buffer
        """
        if size > len(self.experiences):
            experiences_idx = np.random.choice(len(self.experiences), size=size)
        else:
            experiences_idx = np.random.choice(len(self.experiences), size=size, replace=False)

        # from all sampled experiences, extract a tuple of (s,a,r,s')
        states = []
        actions = []
        rewards = []
        states_prime = []
        terminal = []
        for i in range(size):
            states.append(self.experiences[experiences_idx[i]][0])
            actions.append(self.experiences[experiences_idx[i]][1])
            rewards.append(self.experiences[experiences_idx[i]][2])
            states_prime.append(self.experiences[experiences_idx[i]][3])
            terminal.append(self.experiences[experiences_idx[i]][4])

        return states, actions, rewards, states_prime, terminal


buffer = Replay_buffer()






def preprocess_screen(screen):
    screen = skimage.transform.resize(screen, [IMG_WIDTH, IMG_HEIGHT, 1])
    return screen


def stack_frames(input_frames):
    if(len(input_frames) == 1):
        state = np.concatenate(input_frames*4, axis=-1)
    elif(len(input_frames) == 2):
        state = np.concatenate(input_frames[0:1]*2 + input_frames[1:]*2, axis=-1)
    elif(len(input_frames) == 3):
        state = np.concatenate(input_frames + input_frames[2:], axis=-1)
    else:
        state = np.concatenate(input_frames[-4:], axis=-1)

    return state
    


update_every_iteration = 1000
print_every_episode = 500
save_video_every_episode = 5000
NUM_EPISODE = 20000
NUM_EXPLORE = 20
BATCH_SIZE = 32

iter_num = 0





for episode in range(0, NUM_EPISODE + 1):

    # Reset the environment
    env.reset_game()

  
    # input frame
    input_frames = [preprocess_screen(env.getScreenGrayscale())]

    # for every 500 episodes, shutdown exploration to see the performance of greedy action
    if episode % print_every_episode == 0:
        online_agent.shutdown_explore()

    # cumulate reward for this episode
    cum_reward = 0

    t = 0
    while not env.game_over():

        state = stack_frames(input_frames)

        # feed current state and select an action
        action = online_agent.select_action(state)

        # execute the action and get reward
        reward = env.act(env.getActionSet()[action])

      
        # record input frame
        input_frames.append(preprocess_screen(env.getScreenGrayscale()))

        # cumulate reward
        cum_reward += reward

        # observe the result
        state_prime = stack_frames(input_frames)  # get next state

        # append experience for this episode
        if episode % print_every_episode != 0:
            buffer.add((state, action, reward, state_prime, env.game_over()))

        # Setting up for the next iteration
        state = state_prime
        t += 1

        # update agent
        if episode > NUM_EXPLORE and episode % print_every_episode != 0:
            iter_num += 1
            train_states, train_actions, train_rewards, train_states_prime, terminal = buffer.sample(BATCH_SIZE)
            train_states = np.asarray(train_states).reshape(-1, IMG_WIDTH, IMG_HEIGHT, NUM_STACK)
            train_states_prime = np.asarray(train_states_prime).reshape(-1, IMG_WIDTH, IMG_HEIGHT, NUM_STACK)

            # convert Python object to Tensor to prevent graph re-tracing
            train_states = tf.convert_to_tensor(train_states, tf.float32)
            train_actions = tf.convert_to_tensor(train_actions, tf.int32)
            train_rewards = tf.convert_to_tensor(train_rewards, tf.float32)
            train_states_prime = tf.convert_to_tensor(train_states_prime, tf.float32)
            terminal = tf.convert_to_tensor(terminal, tf.bool)

            train_step(train_states, train_actions, train_rewards, train_states_prime, terminal)

        # synchronize target model's weight with online model's weight every 1000 iterations
        if iter_num % update_every_iteration == 0 and episode > NUM_EXPLORE and episode % print_every_episode != 0:
            target_agent.model.set_weights(online_agent.model.get_weights())




    # update exploring rate
    online_agent.update_parameters(episode)
    target_agent.update_parameters(episode)

    if episode % print_every_episode == 0 and episode > NUM_EXPLORE:
        print(
            "[{}] time live:{}, cumulated reward: {}, exploring rate: {}, average loss: {}".
            format(episode, t, cum_reward, online_agent.exploring_rate, average_loss.result()))
        average_loss.reset_states()



#########################################




 

# print(f'list(para.img_shape): {list(para.img_shape)}')
class Agent:

    def __init__(self, name, para):  
        self.model = Agent.build_model(name)
        self.para = deepcopy(para)
        

    @staticmethod
    def build_model(name):
        # input: state
        # output: each action's Q-value
        screen_stack = tf.keras.Input(shape=[self.para.img_shape[1], self.para.img_shape[0], 
                                                        self.para.img_stack_num], dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)
        Q = tf.keras.layers.Dense(self.para.action_num)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=Q)

        return model

    def loss(self, state, action, reward, tar_Q, ternimal):
        output = self.model(state)
        index = tf.stack([tf.range(tf.shape(action)[0]), action], axis=1)
        Q = tf.gather_nd(output, index)
        tar_Q *= ~np.array(terminal)
        loss = tf.reduce_mean(tf.square(reward + self.para.discount_factor * tar_Q - Q))
        return loss

    def max_Q(self, state):
        output = self.model(state)
        return tf.reduce_max(output, axis=1)

    def select_action(self, state): 
        if np.random.rand() < self.para.exploring_rate:
            action = np.random.choice(self.para.num_action)   
        else:
            state = np.expand_dims(state, axis = 0)
            output = self.model(state)
            action = tf.argmax(output, axis=1)[0]

        return action


    def update_parameters(self, episode):
        self.para.exploring_rate = max(self.para.min_exploring_rate, min(0.5, 0.99**((episode) / 30)))

    def shutdown_explore(self):
        # make action selection greedy
        self.para.exploring_rate = 0



    def save_checkpoint(self, path):  
        print(f'saved ckpt {path}') 
        self.model.save_weights(path)
         
    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        print(f'loaded ckpt {path}') 
        self.model(tf.random.uniform(shape=[self.para.img_shape[1], self.para.img_shape[0], 
                                                        self.para.img_stack_num]))
        self.model.load_weights(path)




class Replay_buffer():

    def __init__(self, buffer_size=50000):
        self.experiences = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.experiences) >= self.buffer_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, size):
     
        if size > len(self.experiences):
            idx = np.random.choice(len(self.experiences), size=size)
        else:
            idx = np.random.choice(len(self.experiences), size=size, replace=False)

        # from all sampled experiences, extract a tuple of (s,a,r,s')
        states = []
        actions = []
        rewards = []
        states_prime = []
        terminal = []
        for i in range(size):
            exp = self.experiences[idx[i]]
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            states_prime.append(exp[3])
            terminal.append(exp[4])

        return states, actions, rewards, states_prime, terminal








def rgb2gray(rgb): 
    # TODO: test correctness.
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def preprocess_screen(screen): 
    screen = rgb2gray(screen)
    print(f'screen.shape: {screen.shape}')
    return screen



 

class Trainer():

    def __init__(self, para, buffer):

        self.para = para
        self.buffer = buffer 
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
      
        self.online_agent = Agent('online') 
        self.target_agent = Agent('target') 
        self.target_agent.model.set_weights(self.online_agent.model.get_weights())
 

    def collect_data(self):

        for episode in range(1, self.para.episode_num + 1):

            state = env.reset()       
            input_frames = [preprocess_screen(state)]
            state_stack = stack_frames(input_frames)

            cum_reward = 0

            t = 0
            done = False
            while not done:
  
                action = online_agent.select_action(state_stack)
 
                state_next, reward, done, info = env.step(action)
                cum_reward += reward
             
                input_frames.append(preprocess_screen(state_next))
                state_next_stack = stack_frames(input_frames)  # get next state
 
                self.buffer.add((state_stack, action, reward, state_next_stack, done))
 
                state_stack = state_next_stack
                t += 1



    def train(self):
        for i in range(self.para.iter_num):

            self.collect_data()  

            losses = []
            n = int(len(self.buffer.experiences) / self.para.batch_size)
            for j in range(n):
                experience = self.buffer.sample(self.para.batch_size)
                loss = self.train_step(experience)
                losses.append(loss)

            print(f'iter: {i}, loss: {np.mean(losses)}')

            if i % update_target_agent_period == 0:
                self.target_agent.model.set_weights(self.online_agent.model.get_weights())

            if i % save_period == 0:
                self.online_agent.save_checkpoint(self.path.ckpt + 'checkpoint_{i}.h5')
 

    def train_step(experience):

        states, actions, rewards, states_next, terminal = experience      
        
        states = np.asarray(states).reshape(-1, self.img_shape[1], self.img_shape[0], self.img_stack_num)
        states_next = np.asarray(states_next).reshape(-1, self.img_shape[1], self.img_shape[0], self.img_stack_num)
    
        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        states_next = tf.convert_to_tensor(states_next, tf.float32)
        terminals = tf.convert_to_tensor(terminal, tf.bool)

        loss = self._train_step(states, actions, rewards, states_next, terminals)
        return loss.numpy()


    @tf.function
    def _train_step(state, action, reward, next_state, terminal):
        
        tar_Q = self.target_agent.max_Q(next_state)

        with tf.GradientTape() as tape:
            loss = self.online_agent.loss(state, action, reward, tar_Q, terminal)

        gradients = tape.gradient(loss, self.online_agent.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_agent.model.trainable_variables))
    
    return loss


    def evaluate(self):






para = AttrDict({
    'action_num': len(COMPLEX_MOVEMENT), 
    'img_shape': (240, 256, 3),
    'img_stack_num': 4,

    'buf_size': 2**16,
    
    'iter_num': 20000,
    'episode_num': 10, 
    'batch_size': 32

    'min_exploring_rate': 0.01,

    'discount_factor': 0.99,
    'exploring_rate': 0.1,

    'ckpt': "111022533"
})



buffer = Replay_buffer(para.buf_size)
trainer = Trainer(para, buffer)
trainer.train()




