 
import numpy as np   
import tensorflow as tf 
from PIL import Image


class AttrDict(dict):
    def __getattr__(self, a):
        return self[a]
 

para = AttrDict({
    'action_num': 12, 
    'img_shape': (84, 84, 3),
    'k': 4, 
    'skip': 4
})


def preprocess_screen(screen): 

    def rgb2gray(rgb):  
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    img = Image.fromarray(screen)
    img = img.resize((84, 84), Image.BILINEAR)
    img = np.array(img) # (84, 84, 3)

    img = rgb2gray(img) 
    img = img[..., np.newaxis] # shape is (h, w, 1)
    
    return img




class Agent:

    def __init__(self):   
        # para = para 
        self.model = self.build_model('online')
     
        self.load_checkpoint('./111022533_hw2_data')
 

        self.skip = para.skip  
        self.i = para.skip
        self.prev_action = 1
        self.recent_frames = []
 
 
    def build_model(self, name):
        # input: state
        # output: each action's Q-value
        input_shape = [para.img_shape[0], para.img_shape[1], para.k]
        screen_stack = tf.keras.Input(shape=input_shape, dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack) # (4, 8, 8, 32)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x) # (32, 4, 4, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x) # (64, 3, 3, 64)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(units=512)(x)
        x = tf.keras.layers.ReLU()(x)
        

        adv = tf.keras.layers.Dense(para.action_num)(x)
        v = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=[adv, v])

        return model
 
    def q(self, state):
        adv, v = self.model(state)
        q = v + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
        return q
 
    def select_action(self, state):  

        # state = np.expand_dims(state, axis = 0)
        q = self.q(state)        

        action = tf.argmax(q, axis=1)[0]
        action = int(action.numpy())

        return action
 

    def save_checkpoint(self, path):  
        self.model.save_weights(path)
         
    def load_checkpoint(self, path): 
        self.model(tf.random.uniform(shape=[1, para.img_shape[0], para.img_shape[1], 
                                                        para.k]))
        self.model.load_weights(path)


    def act(self, obs):
        
        if(self.i >= self.skip):

            self.i = 1

            if(len(self.recent_frames) >= para.k): self.recent_frames.pop(0)
            self.recent_frames.append(preprocess_screen(obs))
 
            if  np.random.rand() < 0.01:
                action = np.random.choice(para.action_num)
            else:
                d = len(self.recent_frames)
                state = np.concatenate([np.zeros_like(self.recent_frames[0])[np.newaxis,...]]*(para.k-d)  + [i[np.newaxis,...] for i in self.recent_frames], axis=3)
                assert state.shape == (1, para.img_shape[0], para.img_shape[1], para.k)            
                action = self.select_action(state / 255.0)

            self.prev_action = action

            return action

        else:
            self.i += 1
            return self.prev_action
