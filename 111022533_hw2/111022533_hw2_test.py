 
import numpy as np   
import tensorflow as tf 




class Agent:

    def __init__(self, name, para):   
        self.para = para 
        self.model = self.build_model(name)
     
        self.load_checkpoint('./111022533_hw2_data')
 
    def build_model(self, name):
        # input: state
        # output: each action's Q-value
        input_shape = [self.para.img_shape[0], self.para.img_shape[1], self.para.k]
        screen_stack = tf.keras.Input(shape=input_shape, dtype=tf.float32)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4)(screen_stack) # (4, 8, 8, 32)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2)(x) # (32, 4, 4, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(x) # (64, 3, 3, 64)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=256)(x)
        x = tf.keras.layers.ReLU()(x)
        Q = tf.keras.layers.Dense(self.para.action_num)(x)

        model = tf.keras.Model(name=name, inputs=screen_stack, outputs=Q)

        return model
 
    def max_Q(self, state):
        output = self.model(state)
        return tf.reduce_max(output, axis=1)
 

    def select_action(self, state):  

        # state = np.expand_dims(state, axis = 0)
        output = self.model(state)
        action = tf.argmax(output, axis=1)[0]
        action = int(action.numpy())

        return action

    def save_checkpoint(self, path):  
        # print(f'- saved ckpt {path}') 
        self.model.save_weights(path)
         
    def load_checkpoint(self, path): 
        # need call once to enable load weights.
        # print(f'- loaded ckpt {path}') 
        self.model(tf.random.uniform(shape=[1, self.para.img_shape[0], self.para.img_shape[1], 
                                                        self.para.k]))
        self.model.load_weights(path)



    def act(self, obs):

        def rgb2gray(rgb):  
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

        def preprocess_screen(screen): 
            screen = screen[::2,::2,:]
            screen = rgb2gray(screen) 
            screen = screen[..., np.newaxis] # shape is (h, w, 1)
            return screen
 
        state = np.concatenate([preprocess_screen(obs)] * 4, axis=-1)
        state = np.expand_dims(state, axis = 0)

        assert state.shape == (1, para.frame_shape[0], para.frame_shape[1], para.k)

        return self.select_action(state)
 