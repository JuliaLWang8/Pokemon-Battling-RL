# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import MiniMax

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

def dqn_eval_steps(player, dqn, nb_steps):
    dqn.test(player, nb_episodes=nb_steps, visualize=False, verbose=False)
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

    ### Experiment 2: Effects of changing training opponent
    ## Creating the 3 agents to be trained against different opponents
    agent1 = RLAgent(battle_format="gen8randombattle")
    agent2 = RLAgent(battle_format="gen8randombattle")
    agent2a = RLAgent(battle_format="gen8randombattle")
    agent3 = RLAgent(battle_format="gen8randombattle")
    agent4 = RLAgent(battle_format="gen8randombattle")
    agent5 = RLAgent(battle_format="gen8randombattle")

    # Opponents:
    opponent = RandomPlayer(battle_format="gen8randombattle")  # to train agent1
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle") # to train agent 2
    third_opponent = MiniMax.MinimaxPlayer(battle_format="gen8randombattle")
	# agent3 trained against another RL agent

    ## Output dimensions
    n_action1 = len(agent1.action_space)
    n_action2 = len(agent2.action_space)
    n_action2a = len(agent4.action_space)
    n_action3 = len(agent3.action_space)
    n_action4 = len(agent4.action_space)
    n_action5 = len(agent4.action_space)

    # setting up model for agent 1
    model1 = Sequential()
    model1.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model1.add(Flatten())
    model1.add(Dense(64, activation="elu"))
    model1.add(Dense(n_action1, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    ### Creating policy (same across 3 agents)
    policy1 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )
    policy2 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )
    policy2a = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )
    policy3 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )
    policy4 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )
    policy5 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )
    # model1 = tf.keras.models.load_model('model_1_10000')

    ### Agent 1
    ## Defining DQN for agent 1
    dqn1 = DQNAgent(
        model=model1,
        nb_actions=len(agent1.action_space),
        policy=policy1,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    
    # Compile dqn using the Adam optimizer
    dqn1.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    
    ## Training agent 1
	# plays against random player
    agent1.play_against( env_algorithm=dqn_training, opponent=opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_steps": NB_TRAINING_STEPS},)
    model1.save(f"model_1_{NB_TRAINING_STEPS}")

    ## Evaluating agent 1
    print("Agent 1 results against random player:")
    agent1.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 1 results against max player:")
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 1 results against minimax player:")
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
    
    ### Agent 2
    ## Setting up model
    model2 = Sequential()
    model2.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model2.add(Flatten())
    model2.add(Dense(64, activation="elu"))
    model2.add(Dense(n_action2, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model2 = tf.keras.models.load_model('model_2_10000')

    ## defining dqn for agent 2
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
    dqn2.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    ## Training agent 2
	# playes against MaxDamagePlayer
    agent2.play_against( env_algorithm=dqn_training, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_steps": NB_TRAINING_STEPS},)
    model2.save(f"model_2_{NB_TRAINING_STEPS}")

    ## Evaluating agent 2
    print("Agent 2 results against random player:")
    agent2.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 2 results against max player:")
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 2 results against minimax player:")
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES},)

    ### Agent 2a
    ## Setting up model
    model2a = Sequential()
    model2a.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model2a.add(Flatten())
    model2a.add(Dense(64, activation="elu"))
    model2a.add(Dense(n_action2a, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model2 = tf.keras.models.load_model('model_2a_10000')

    ## defining dqn for agent 2a
    dqn2a = DQNAgent(
        model=model2a,
        nb_actions=len(agent2a.action_space),
        policy=policy2a,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn2a.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    ## Training agent 2a
	# playes against MiniMaxPlayer
    agent2a.play_against( env_algorithm=dqn_training, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn2a, "nb_steps": NB_TRAINING_STEPS},)
    model2a.save(f"model_2a_{NB_TRAINING_STEPS}")

    ## Evaluating agent 2a
    print("Agent 2a results against random player:")
    agent2a.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn2a, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 2a results against max player:")
    agent2a.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn2a, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 2a results against minimax player:")
    agent2a.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn2a, "nb_episodes": NB_EVALUATION_EPISODES},)

    ### Agent 3
    #Setting up model
    model3 = Sequential()
    model3.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model3.add(Flatten())
    model3.add(Dense(64, activation="elu"))
    model3.add(Dense(n_action3, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model3 = tf.keras.models.load_model('model_3_10000')

    # defining dqn for agent 3    
    dqn3 = DQNAgent(
        model=model3,
        nb_actions=len(agent3.action_space),
        policy=policy3,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn3.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    ## Training agent 3
	# plays against other RL agent
    agent3.play_against( env_algorithm=dqn_training, opponent=agent1, env_algorithm_kwargs={"dqn": dqn3, "nb_steps": 2*NB_TRAINING_STEPS}, env_algorithm2=dqn_eval_steps,)
    model3.save(f"model_3_{NB_TRAINING_STEPS}")

    # Evaluating agent 3
    print("Agent 3 results against random player:")
    agent3.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 3 results against max player:")
    agent3.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 3 results against minimax player:")
    agent3.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES},)

    ### Agent 4
    #Setting up model
    model4 = Sequential()
    model4.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model4.add(Flatten())
    model4.add(Dense(64, activation="elu"))
    model4.add(Dense(n_action4, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model4 = tf.keras.models.load_model('model_4_10000')

    # defining dqn for agent 4    
    dqn4 = DQNAgent(
        model=model4,
        nb_actions=len(agent4.action_space),
        policy=policy4,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn4.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    ## Training agent 4
	# plays against other RL agent
    agent4.play_against( env_algorithm=dqn_training, opponent=agent2, env_algorithm_kwargs={"dqn": dqn4, "nb_steps": 2*NB_TRAINING_STEPS}, env_algorithm2=dqn_eval_steps,)
    model4.save(f"model_4_{NB_TRAINING_STEPS}")

    # Evaluating agent 4
    print("Agent 4 results against random player:")
    agent4.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn4, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 4 results against max player:")
    agent4.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn4, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 4 results against minimax player:")
    agent4.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn4, "nb_episodes": NB_EVALUATION_EPISODES},)

    ### Agent 5
    #Setting up model
    model5 = Sequential()
    model5.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model5.add(Flatten())
    model5.add(Dense(64, activation="elu"))
    model5.add(Dense(n_action5, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model5 = tf.keras.models.load_model('model_5_10000')

    # defining dqn for agent 5    
    dqn5 = DQNAgent(
        model=model5,
        nb_actions=len(agent4.action_space),
        policy=policy5,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn5.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    ## Training agent 5
	# plays against other RL agent
    agent5.play_against( env_algorithm=dqn_training, opponent=agent2a, env_algorithm_kwargs={"dqn": dqn5, "nb_steps": 2*NB_TRAINING_STEPS}, env_algorithm2=dqn_eval_steps,)
    model5.save(f"model_5_{NB_TRAINING_STEPS}")

    # Evaluating agent 4
    print("Agent 5 results against random player:")
    agent5.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn5, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 5 results against max player:")
    agent5.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn5, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 5 results against minimax player:")
    agent5.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn5, "nb_episodes": NB_EVALUATION_EPISODES},)

    ## Round Robin
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent2, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent2a, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent3, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent4, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent5, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)

    agent2.play_against(env_algorithm=dqn_evaluation, opponent=agent2a, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=agent3, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=agent4, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=agent5, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)

    agent2a.play_against(env_algorithm=dqn_evaluation, opponent=agent3, env_algorithm_kwargs={"dqn": dqn2a, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent2a.play_against(env_algorithm=dqn_evaluation, opponent=agent4, env_algorithm_kwargs={"dqn": dqn2a, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent2a.play_against(env_algorithm=dqn_evaluation, opponent=agent5, env_algorithm_kwargs={"dqn": dqn2a, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)

    agent3.play_against(env_algorithm=dqn_evaluation, opponent=agent4, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent3.play_against(env_algorithm=dqn_evaluation, opponent=agent5, env_algorithm_kwargs={"dqn": dqn3, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    
    agent4.play_against(env_algorithm=dqn_evaluation, opponent=agent5, env_algorithm_kwargs={"dqn": dqn4, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)