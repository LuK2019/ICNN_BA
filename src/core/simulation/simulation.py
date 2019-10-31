import numpy as np 
import pandas as pd
import tensorflow as tf

from ..optimization.bundle_entropy import BundleEntropyMethod

def H(y,eps=0.0001): #TODO: Test this
    y1,y2 = y
    y = np.array([y1, y2])
    # To enforce avoiding log(0), TODO:test this
    k = 0
    for element in [y1, y2]:
        if element > 1-eps:
            y[k] = 0.99
        if element < eps:
            y[k] = 0.01
        k += 1
    return tf.Variable(-np.sum(y*np.log(y) + (1.-y)*np.log(1.-y)), dtype="float32")

def parse_4d_state(current_state):
    a,b,c,d = current_state
    return np.array([[a],[b],[c],[d]])

def parse_2d_action(current_action):
    a,b = current_action
    return np.array([[a],[b]])



class simulation:
    def __init__(self, ICNN_model, game, num_episodes, size_minibatches,\
         capacity_replay_memory, initial_weight_vector=None,\
              moving_average_factor=0.01, greedy_factor=0.05,\
                   discount_factor=0.99, optimization_iterations=5, initial_action_for_optimization=np.array([0.5, 0.5]), update_target_frequency=1.):
        self.negQ = ICNN_model
        self.negQ_target = ICNN_model
        self.num_episodes = num_episodes
        self.size_minibatches = size_minibatches
        self.capacity_replay_memory = capacity_replay_memory
        self.initial_weight_vector = initial_weight_vector
        self.moving_average_factor = moving_average_factor 
        self.greedy_factor = greedy_factor
        self.discount_factor = discount_factor
        self.optimization_iterations = optimization_iterations
        self.game = game
        self.update_target_frequency = update_target_frequency
        self.initial_action_for_optimization = initial_action_for_optimization

    def run_simulation(self):
        replay_memory = pd.DataFrame(columns=["current_state", "action", "reward", "next_state"])
        # Convenvtion used: some_entry = {"current_state":(1,2,3,4): , "action":(0.5, 0.3), "reward": 4, "next_state": (1,3,5,6)} -> all entries are python/numpy objects NOT tf.Variables/Tensors

        for episode in np.arange(self.num_episodes):
            current_state = (self.game.x_0, self.game.y_0, self.game.S_0, 0)
            for period in np.arange(self.game.T-1):
                print("We are in episode {}/{} and period {}/{}, {}% progress".format(episode, self.num_episodes, period, self.game.T, (period*self.game.T+period)/(self.num_episodes*self.game.T)))
                # Check for exploration action #TODO: Add some decay of greedy rate 
                if np.random.binomial(1,self.greedy_factor)==1:
                    action = np.array([np.random.uniform(0,1), np.random.uniform(0,1)])
                    print("GREEDY: Random action {} chosen".format(action))
                else:
                    action = BundleEntropyMethod(self.negQ,\
                         parse_4d_state(current_state),\
                              parse_2d_action(self.initial_action_for_optimization),\
                                   K=self.optimization_iterations) #TODO: Do something about the sigular matrix thing
                # Execute action and watch environment
                transition = self.game.get_new_state(current_state, action)

                # Store transition in replay memory
                replay_memory = replay_memory.append(transition, ignore_index=True)

                # TODO: Here one would need to remove superflous entries in replay memory

                # Sample random_minibatch
                random_minibatch = replay_memory.sample(self.size_minibatches, replace=True) #TODO: Check if there is a chance to do this without the replace thing
                # Add empty "y_m" column to store y_m
                random_minibatch = random_minibatch.reindex(columns=["current_state", "action", "reward", "next_state","y_m"])
                for index, transition in random_minibatch.iterrows():
                    action_plus = BundleEntropyMethod(self.negQ, parse_4d_state(transition["next_state"]),\
                        parse_2d_action(self.initial_action_for_optimization),\
                             K=self.optimization_iterations ) #TODO: Parse the arguments and the return value to tf.Tensor

                    # Determine y_m , this entails checking if the episode ends now
                    if episode == self.game.T - 1:
                        y_m = transition["reward"]
                    else:
                        y_m = transition["reward"]\
                            + self.discount_factor*\
                                self.negQ_target((parse_4d_state(transition["next_state"]),action_plus))
                    # Add y_m at the corresponding location
                    random_minibatch.iloc[index, 4] = y_m.numpy()[0][0]
                    # Fill up the missing "y_m" entries 
                    if self.size_minibatches > episode:
                        random_minibatch = random_minibatch.fillna(y_m.numpy()[0][0]) 

                # Prepare Data for Loss Calculation
                negQ_input = [(tf.Variable(parse_4d_state(transition_mini["current_state"]), dtype="float32"), tf.Variable(parse_2d_action(transition_mini["action"]), dtype="float32")) for index,transition_mini in random_minibatch.iterrows() ]

                
                # Calculate -H(a_m) + y_m vectorized, with changed signs, b.c. it will be the input to the MSE keras function #TODO: Check if this data has to be tf.Variable
                y_true_H = np.array([[H(transition_mini["action"]) for index,transition_mini in random_minibatch.iterrows()]]) 
                y_true_y_m = np.array([[transition_mini["y_m"] for index, transition_mini in random_minibatch.iterrows()]])
                y_true = -y_true_H + y_true_y_m
                y_true = tf.Variable(y_true, dtype="float32")
                y_true = tf.reshape(y_true, [y_true.shape[1],])


                with tf.GradientTape() as tape:
                     y_pred = tf.Variable([self.negQ(arg)[0][0] for arg in negQ_input])
                     loss = tf.keras.losses.MSE(y_true, y_pred)



                    # first_vec = \
                    #     [self.negQ((parse_4d_state(transition_mini["current_state"]), parse_2d_action(transition_mini["action"])))\
                    #             for index, transition_mini in random_minibatch.iterrows()]
                    # second_vec = [H(transition_mini["action"]) for index, transition_mini in random_minibatch.iterrows()]
                    # third_vec = [tf.Variable(transition_mini["y_m"], dtype="float32") for index, transition_mini in random_minibatch.iterrows()]
                    # loss = tf.reduce_mean(tf.pow((first_vec+second_vec-third_vec), tf.Variable(2., dtype="float32")))




                gradient = tape.gradient(loss, self.negQ.trainable_variables)
                # Gradient step #TODO: Do the real thing
                keras.optimizers.Nadam(lr=self.moving_average_factor).apply_gradients(zip(gradients, self.negQ.trainable_variables))
                print("Current loss", keras.metrics.Mean(loss))

                # Apply the restrictions to the weights
                for variable in self.negQ.variables:
                    if variable.constraint is not None:
                        variable.assgn(variable.constraint(variable))

                # Update target network parameters if necessary
                if period%self.update_target_frequency==0:
                    self.negQ_target = self.negQ
                # Update the next current state
                current_state=tramsition["next_state"]

if __name__=="__main__":
    import os
    print(os.chdir(r"C:\Users\lukas\OneDrive\Universität\Mathematik\Bachelorarbeit\dev\src"))
    print(os.getcwd())

                
    
