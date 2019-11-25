import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from ..optimization.gradient_descent import GradientDescent
from ..optimization.bundle_entropy import BundleEntropyMethod
from . import validation
from ..utils.utils import (
    H,
    parse_4d_state,
    parse_2d_action,
    create_targets,
    check_model_input,
    create_arguments,
    greedy_estimator,
    grad,
    plot_function,
)


class simulation:
    def __init__(
        self,
        ICNN_model,
        game,
        num_episodes,
        size_minibatches,
        capacity_replay_memory,
        show_plot_every,
        LOG_NUM,
        initial_weight_vector=None,
        ITERATIONS=10,
        moving_average_factor=0.01,
        greedy_factor=greedy_estimator,
        discount_factor=0.99,
        optimization_iterations=5,
        initial_action_for_optimization=tf.Variable([[0.2], [0.7]], dtype="float32"),
        update_target_frequency=1.0,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        LOG_PATH=r"C:\Users\lukas\OneDrive\Universität\Mathematik\Bachelorarbeit\log_dir",
    ):
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
        self.ITERATIONS = ITERATIONS
        self.optimizer = optimizer
        self.show_plot_every = show_plot_every
        self.LOG_PATH = LOG_PATH
        self.LOG_NUM = LOG_NUM
        self.check_point_amount = 3

    def run_simulation(self):
        """Conventions for replay_memory:
        {
            "current_state": np.ndarray [4,1] np.float32,
            "action": np.ndarray [2,1] np.float32,
            "reward": float/np.ndarray [1,] TODO: TBD
            "next_state": np.ndarray [4,1] np.float32
        }


        """
        # Create logging directory
        LOG_DIR = os.path.join(self.LOG_PATH, "log_{}".format(self.LOG_NUM))
        LOG_DIR_WEIGHTS = os.path.join(LOG_DIR, "WEIGHTS")
        LOG_DIR_SIMULATION_SUMMARY = os.path.join(LOG_DIR, "SIMULATION_SUMMARY")
        print("The Log of this training session is @:", LOG_DIR)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
            os.makedirs(LOG_DIR_WEIGHTS)
            os.makedirs(LOG_DIR_SIMULATION_SUMMARY)
        else:
            raise OSError(
                "There already exists a logging under {}, change LOG_NUM to a unused logging directory".format(
                    LOG_DIR
                )
            )
        # Initialize logging data

        simulation_summary = pd.DataFrame(
            columns=[
                "episode",
                "greedy",
                "agent_choice_of_x1",
                "optimal_choice_of_x1",
                "agent_final_cash_balance",
                "optimal_final_cash_balance",
                "cumulative_liquidity_costs",
            ]
        )
        # Initialize replay_memory
        replay_memory = pd.DataFrame(
            columns=["current_state", "action", "reward", "next_state"]
        )
        # Convenvtion used: some_entry = {"current_state":(1,2,3,4): , "action":np.ndarray([[0.5], [0.5]]), "reward": 4, "next_state": (1,3,5,6)} -> all entries are python/numpy objects NOT tf.Variables/Tensors
        checkpoint_counter = 1
        for episode in np.arange(self.num_episodes):
            # The results of each episode are recorded. At the end of the episode they are appended to the simulation_summary
            episode_summary = {"greedy": [0]}
            episode_summary["episode"] = [episode]
            # Model weights are saved periodically
            if episode % int(self.num_episodes / self.check_point_amount) == 0:
                print("Weights are saved for the {}. time".format(checkpoint_counter))
                WEIGHT_DIR = os.path.join(LOG_DIR_WEIGHTS, "1")
                self.negQ.save_weights(WEIGHT_DIR)
                checkpoint_counter += 1

            current_state = np.array(
                [[self.game.x_0], [self.game.y_0], [self.game.S_0], [0]],
                dtype=np.float32,
            )
            episode_summary["initial_state"] = [current_state]
            cumulative_liquidity_costs = 0
            for period in np.arange(self.game.T):
                # Print current status
                print(
                    "We are in episode {}/{} and period {}/{}, {}% progress".format(
                        episode,
                        self.num_episodes - 1,
                        period,
                        self.game.T - 1,
                        (episode / self.num_episodes) * 100,
                    )
                )

                # Check for exploration action
                if (
                    np.random.binomial(
                        1,
                        self.greedy_factor(self.num_episodes, episode, 0.05, 0.1, 0.2),
                    )
                    == 1
                ):
                    action = np.array(
                        [[np.random.uniform(0.1, 0.9)], [np.random.uniform(0.1, 0.9)]],
                        dtype=np.float32,
                    )
                    episode_summary["greedy"] = [1]
                    print(
                        "\n",
                        "\n===",
                        "GREEDY: Random action {} chosen at greedy factor {}".format(
                            action,
                            self.greedy_factor(
                                self.num_episodes, episode, 0.05, 0.1, 0.5
                            ),
                        ),
                    )
                else:
                    action = BundleEntropyMethod(
                        self.negQ,
                        current_state,
                        self.initial_action_for_optimization,
                        K=self.optimization_iterations,
                    )

                # Execute action and watch environment
                transition, liquidity_costs = self.game.get_new_state_adjusted(
                    current_state, action
                )  # TODO: Remove old get_new_state and rename it
                cumulative_liquidity_costs += liquidity_costs
                assert (
                    transition["next_state"][3, 0] <= self.game.T
                ), "We have a period which is larger than {} at transition: {}".format(
                    self.game.T, transition
                )

                # Store action
                if period == 0:  # TODO: Adjust for multi period
                    chosen_x1 = transition["next_state"][0, 0]
                    optimal_x1 = validation.optimum_2p_solution(
                        current_state, self.game
                    )
                    print(
                        "\n===",
                        "Your choice: {}, Optimal choice: {}, Difference: {}".format(
                            chosen_x1, optimal_x1, chosen_x1 - optimal_x1
                        ),
                        "\n===",
                    )
                    episode_summary["agent_choice_of_x1"] = [chosen_x1]
                    episode_summary["optimal_choice_of_x1"] = [optimal_x1]

                # Store transition in replay memory
                replay_memory = replay_memory.append(transition, ignore_index=True)

                # If replay_memory has more than self.capacity_replay_memory entries, only keep the newest self.capacity_replay_memory entries in replay_memory
                if replay_memory.shape[0] > self.capacity_replay_memory:
                    replay_memory = replay_memory[-self.capacity_replay_memory :]

                # Sample random_minibatch
                random_minibatch = replay_memory.sample(
                    self.size_minibatches, replace=True
                )  # TODO: Check if there is a chance to do this without the replace thing
                # Add empty "y_m" column to store y_m
                random_minibatch = random_minibatch.reindex(
                    columns=["current_state", "action", "reward", "next_state", "y_m"]
                )
                # Give the random_minibatch an index from 0,1,...,self.size_minibatches
                random_minibatch.index = list(np.arange(random_minibatch.shape[0]))

                # The network is trained with the loss from the data in the minibatch
                for index, transition_batch in random_minibatch.iterrows():
                    # For each transition in our minibatch, we
                    # 1.Determine the optimal action (action_plus) of our target_network at "next_state"
                    action_plus = BundleEntropyMethod(
                        self.negQ_target,
                        transition_batch["next_state"],
                        self.initial_action_for_optimization,
                        K=self.optimization_iterations,
                    )

                    # 2.Determine the total reward, (with or without Q_target) y_m = r_m + gamma*Q_target("next_state", action_plus)
                    # Case 2.1: Transition from the penultimate period (period==T-1) to the final period at which the episode ends
                    if transition_batch["current_state"][3] == (self.game.T - 1):
                        y_m = transition_batch[
                            "reward"
                        ]  # TODO: TBD needs to have same dtype as reward
                    # Case 2.2: Transition from state_t to state_(t+1), whereas state_(t+1) is not the final period
                    elif transition_batch["current_state"][3] < (self.game.T - 1):
                        x_arg = transition_batch["next_state"]
                        y_arg = action_plus
                        x_arg = x_arg.reshape((1, 4, 1))
                        y_arg = y_arg.reshape((1, 2, 1))
                        x_arg = tf.convert_to_tensor(x_arg)
                        y_arg = tf.convert_to_tensor(y_arg)
                        argument = (x_arg, y_arg)
                        assert check_model_input(argument)
                        y_m = (
                            transition_batch["reward"]
                            + self.discount_factor * self.negQ_target(argument).numpy()
                        )  # Ensure np.float32
                        y_m = y_m[
                            0, 0, 0
                        ]  # Unpack y_m from [1,1,1] np.array to float32
                    # Add y_m at the corresponding location
                    else:
                        print(
                            "We have a transition from current_state:period {} -> next_state:period {}, which is invalid".format(
                                transition_batch["current_state"][3],
                                transition_batch["next_state"][3],
                            )
                        )
                    try:
                        random_minibatch.iloc[index, 4] = y_m
                    except:
                        print("[index, 4]", [index, 4])
                        print("Data Frame", random_minibatch)
                        raise

                    print(
                        "y_m  from period {} to period {} is {}".format(
                            transition_batch["current_state"][3],
                            transition_batch["next_state"][3],
                            y_m,
                        )
                    )

                    # Fill up the missing "y_m" entries
                    if self.size_minibatches > episode:
                        random_minibatch = random_minibatch.fillna(y_m)

                ####################################################
                ################### TRAINING:START #################
                ####################################################

                # Prepare data for loss calculation
                y_target = create_targets(random_minibatch=random_minibatch)

                # negQ argument:
                argument_training = create_arguments(random_minibatch=random_minibatch)
                assert check_model_input(argument_training)
                # Calculate the inital loss
                loss_before, gradient_before = grad(
                    self.negQ, argument_training, y_target
                )
                if np.isnan(loss_before.numpy()):
                    print(
                        "Argument",
                        argument_training,
                        "y_target",
                        y_target,
                        "grad",
                        grads,
                    )
                # Gradient descent:
                x_axis = list(np.arange(self.ITERATIONS))
                for i in np.arange(self.ITERATIONS):
                    loss_value, grads = grad(self.negQ, argument_training, y_target)
                    # For clipping grads
                    grads_clipped = [
                        tf.clip_by_norm(weight_grad, 15) for weight_grad in grads
                    ]
                    print("Gradient from training", grads[0][0, :])

                    # Application
                    self.optimizer.apply_gradients(
                        zip(grads_clipped, self.negQ.trainable_variables)
                    )

                    loss_value, grads = grad(self.negQ, argument_training, y_target)

                    print(
                        "Loss before: {}, Loss after {}, Loss before - Loss after: {}".format(
                            loss_before, loss_value, loss_before - loss_value
                        )
                    )

                    # Apply the restrictions to the weights
                    for variable in self.negQ.variables:
                        if variable.constraint is not None:
                            variable.assign(variable.constraint(variable))

                    ####################################################
                    ################### TRAINING:END ###################
                    ####################################################
                if (episode % 10 == 0) & (episode > 2):
                    d = {
                        "episode": episode,
                        "weight0": [self.negQ.trainable_variables[0][0][0]],
                        "weight3": [self.negQ.trainable_variables[0][3][0]],
                        "weight10": [self.negQ.trainable_variables[0][10][0]],
                    }

                if (episode % self.show_plot_every == 0) & (episode > 2):
                    print(
                        "At episode {} our current objective (regularized) looks like:".format(
                            episode
                        )
                    )
                    plot_function(
                        self.negQ,
                        argument_training[0][0],
                        GRANULARITY=0.02,
                        regularized=False,
                    )

                # Update target network parameters if necessary
                if period % self.update_target_frequency == 0:
                    self.negQ_target = self.negQ
                # Update the next current state
                current_state = transition["next_state"]
                print(
                    "Period finished, next state will ",
                    "\n==",
                    current_state,
                    "\n===",
                    "The transition of this period was:",
                    "\n====",
                    transition,
                )

                # Store final cash balance
                if period == 1:  # TODO: Adjust for multi period
                    assert (
                        current_state[3, 0] == 2
                    ), "If we are at period == 1, i.e. the final period, the current_state for the next iteration should be prepped as one of period 2"
                    episode_summary["agent_final_cash_balance"] = [
                        transition["next_state"][1, 0]
                    ]
                    # Calculate final cash balance for the optimal x1 choice
                    optimal_final_cash_balance = validation.optimal_final_cash_balance_calc(
                        episode_summary["initial_state"][0],
                        transition["current_state"],
                        episode_summary["optimal_choice_of_x1"][0],
                        self.game,
                    )
                    episode_summary["optimal_final_cash_balance"] = [
                        optimal_final_cash_balance
                    ]
                    episode_summary["cumulative_liquidity_costs"] = [
                        cumulative_liquidity_costs
                    ]
                    episode_summary = pd.DataFrame(episode_summary)
                    simulation_summary = simulation_summary.append(
                        episode_summary, ignore_index=True
                    )
        # Training has finished:
        # Save the experience
        self.replay_memory = replay_memory
        self.simulation_summary = simulation_summary
        simulation_summary.to_pickle(
            os.path.join(
                LOG_DIR_SIMULATION_SUMMARY,
                "simulation_summary_log_{}.pkl".format(self.LOG_NUM),
            )
        )
        WEIGHT_DIR = os.path.join(LOG_DIR_WEIGHTS, "1")
        self.negQ.save_weights(WEIGHT_DIR)
        # END
