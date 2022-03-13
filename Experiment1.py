# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


## RL Agent Class
class RLAgent(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        ## State embedder
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        ## computes reward
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )


class MaxDamagePlayer(RandomPlayer):
    ### chooses the moves with the most damage

    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)


# function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps)
    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


if __name__ == "__main__":

    ### Experiment 1: Effects of changing policies
    ## Creating the 3 agents to be trained with different policies
    agent1 = RLAgent(battle_format="gen8randombattle")
    agent2 = RLAgent(battle_format="gen8randombattle")
    agent3 = RLAgent(battle_format="gen8randombattle")

    # Opponents to train against and test are the same
    opponent = RandomPlayer(battle_format="gen8randombattle")  
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    ## Output dimensions
    n_action1 = len(agent1.action_space)
    n_action2 = len(agent2.action_space)
    n_action3 = len(agent3.action_space)

    # setting up model for agent 1
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action1, activation="linear"))
    memory = SequentialMemory(limit=10000, window_length=1)

    ### Creating policies
    policy1 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), # eps greedy q policy
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )
    policy2 = LinearAnnealedPolicy(
        GreedyQPolicy(),    # greedy q policy
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )
    policy3 = LinearAnnealedPolicy(
        BoltzmannQPolicy(), #boltzmann q policy
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )
    
    ### Agent 1
    # Defining DQN for agent 1
    dqn1 = DQNAgent(
        model=model,
        nb_actions=len(agent1.action_space),
        policy=policy1,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    
    ## Compile dqn using the Adam optimizer
    dqn1.compile(Adam(lr=0.00025), metrics=["mae"])
    
    ## Training agent 1
    agent1.play_against( env_algorithm=dqn_training, opponent=opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_steps": NB_TRAINING_STEPS},)
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluating agent 1
    print("Agent 1 results against random player:")
    agent1.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 1 results against max player:")
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
  
    ### Agent 2
    #Setting up model
    model2 = Sequential()
    model2.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model2.add(Flatten())
    model2.add(Dense(64, activation="elu"))
    model2.add(Dense(n_action2, activation="linear"))
    memory = SequentialMemory(limit=10000, window_length=1)

    # defining dqn for agent 2
    dqn2 = DQNAgent(
        model=model2,
        nb_actions=len(agent2.action_space),
        policy=policy2,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn2.compile(Adam(lr=0.00025), metrics=["mae"])

    # Training agent 2
    agent2.play_against( env_algorithm=dqn_training, opponent=opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_steps": NB_TRAINING_STEPS},)
    model2.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluating agent 2
    print("Agent 2 results against random player:")
    agent2.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 2 results against max player:")
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES},)
  
    ### Agent 3
    #Setting up model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action3, activation="linear"))
    memory = SequentialMemory(limit=10000, window_length=1)

    # defining dqn for agent 3    
    dqn3 = DQNAgent(
        model=model,
        nb_actions=len(agent3.action_space),
        policy=policy3,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn3.compile(Adam(lr=0.00025), metrics=["mae"])

    ## Training agent 3
    agent3.play_against( env_algorithm=dqn_training, opponent=opponent, env_algorithm_kwargs={"dqn": dqn3, "nb_steps": NB_TRAINING_STEPS},)
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluating agent 3
    print("Agent 3 results against random player:")
    agent3.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 3 results against max player:")
    agent3.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES},)
  