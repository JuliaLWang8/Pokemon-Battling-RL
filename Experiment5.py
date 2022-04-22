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
from stable_baselines import PPO2

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
    agent3 = RLAgent(battle_format="gen8randombattle")

    # Opponents:
    opponent = RandomPlayer(battle_format="gen8randombattle")  # to train agent1
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle") # to train agent 2
    third_opponent = MiniMax.MinimaxPlayer(battle_format="gen8randombattle")
	# agent3 trained against another RL agent

    ## Output dimensions
    n_action1 = len(agent1.action_space)
    n_action2 = len(agent2.action_space)
    n_action3 = len(agent3.action_space)

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
    policy3 = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )

    ### Agent 1
    ## Setting up model
    model1 = Sequential()
    model1.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model1.add(Flatten())
    model1.add(Dense(64, activation="elu"))
    model1.add(Dense(n_action1, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model1 = tf.keras.models.load_model('model_1_10000')

    ## defining dqn for agent 1
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
    dqn1.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    ## Training agent 1
	# playes against MiniMaxPlayer
    agent1.play_against( env_algorithm=dqn_training, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_steps": NB_TRAINING_STEPS},)
    model1.save(f"model_1_{NB_TRAINING_STEPS}")

    ## Evaluating agent 1
    print("Agent 1 results against random player:")
    agent1.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 1 results against max player:")
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)
    print("\nAgent 1 results against minimax player:")
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES},)

    ### Agent 5
    #Setting up model
    model2 = Sequential()
    model2.add(Dense(128, activation="elu", input_shape=(1, 10)))
    # embeddings have shape (1, 10), which affects our hidden layer dimension and output dimension
    model2.add(Flatten())
    model2.add(Dense(64, activation="elu"))
    model2.add(Dense(n_action2, activation="linear"))
    memory = SequentialMemory(limit=NB_TRAINING_STEPS, window_length=1)

    # model5 = tf.keras.models.load_model('model_2_10000')

    # defining dqn for agent 5    
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

    ## Training agent 5
	# plays against other RL agent
    agent2.play_against( env_algorithm=dqn_training, opponent=agent1, env_algorithm_kwargs={"dqn": dqn2, "nb_steps": 2*NB_TRAINING_STEPS}, env_algorithm2=dqn_eval_steps,)
    model2.save(f"model_2_{NB_TRAINING_STEPS}")

    # Evaluating agent 4
    print("Agent 5 results against random player:")
    agent2.play_against( env_algorithm=dqn_evaluation, opponent=opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 5 results against max player:")
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=second_opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES})
    print("\nAgent 5 results against minimax player:")
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=third_opponent, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES},)

    model = PPO2("MlpPolicy", agent3, gamma=0.5, verbose=0)

    def ppo_training(player):
        print ("Training...")
        model.learn(total_timesteps=NB_TRAINING_STEPS)
        print("Training complete.")
        
    def ppo_evaluating(player):
        player.reset_battles()
        for _ in range(NB_EVALUATION_EPISODES):
            done = False
            obs = player.reset()
            while not done:
                action = model.predict(obs)[0]
                obs, _, done, _ = player.step(action)
                # print ("done:" + str(done))
        player.complete_current_battle()

        print(
            "PPO Evaluation: %d victories out of %d episodes"
            % (player.n_won_battles, NB_EVALUATION_EPISODES)
        )


    ## Round Robin
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent2, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    agent1.play_against(env_algorithm=dqn_evaluation, opponent=agent3, env_algorithm_kwargs={"dqn": dqn1, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    
    agent2.play_against(env_algorithm=dqn_evaluation, opponent=agent3, env_algorithm_kwargs={"dqn": dqn2, "nb_episodes": NB_EVALUATION_EPISODES}, env_algorithm2=dqn_evaluation,)
    