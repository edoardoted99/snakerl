import os
import csv
import json
from agent import Agent
from game import SnakeGameAI
from helper import plot
import config

def get_latest_train_id(base_path="train_"):
    """ Get the latest training ID by scanning the directory. """
    train_dirs = [d for d in os.listdir('trains') if os.path.isdir(os.path.join('trains', d)) and d.startswith(base_path)]
    train_ids = [int(d.replace(base_path, '')) for d in train_dirs]
    return max(train_ids) if train_ids else 0

def create_train_folder():
    """ Create a new training folder with an incremented ID. """
    latest_id = get_latest_train_id()
    new_id = latest_id + 1
    new_folder = os.path.join("trains", f"train_{new_id}")
    os.makedirs(new_folder)
    return new_folder

def save_episode_info(folder, episode, score, record, total_score, mean_score, epsilon, loss):
    """ Save episode information into a CSV file. """
    csv_file = os.path.join(folder, "train_data.csv")
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Score", "Record", "Total Score", "Mean Score", "Epsilon", "Loss"])  # Header
        writer.writerow([episode, score, record, total_score, mean_score, epsilon, loss])

def save_train_config(folder, config):
    """ Save training configuration details into a JSON file. """
    config_file = os.path.join(folder, "train_config.json")
    with open(config_file, mode='w') as file:
        json.dump(config, file, indent=4)

def train(max_games=config.MAX_GAMES):
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    plot_epsilons = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    train_folder = create_train_folder()

    # Save training configuration
    config_data = {
        "MAX_MEMORY": config.MAX_MEMORY,
        "BATCH_SIZE": config.BATCH_SIZE,
        "LR": config.LR,
        "SPEED": config.SPEED,
        "MAX_GAMES": max_games,
        "REWARD_FRUIT": config.REWARD_FRUIT,
        "REWARD_COLLISION": config.REWARD_COLLISION,
        "REWARD_STEP": config.REWARD_STEP
    }
    save_train_config(train_folder, config_data)

    while agent.n_games < max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(os.path.join(train_folder, 'best_model.pth'))

            total_score += score
            mean_score = total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot_losses.append(loss if loss is not None else 0)
            plot_epsilons.append(agent.epsilon if agent.epsilon is not None else 0)
            
            save_episode_info(train_folder, agent.n_games, score, record, total_score, mean_score, agent.epsilon, loss)
            plot(plot_scores, plot_mean_scores, plot_losses, plot_epsilons, train_folder, agent.n_games, config_data)

            # Print the values to the terminal
            if loss is not None:
                print(f'Game {agent.n_games} Score: {score} Record: {record} Total Score: {total_score} Mean Score: {mean_score:.2f} Epsilon: {agent.epsilon:.2f} Loss: {loss:.2f}')
            else:
                print(f'Game {agent.n_games} Score: {score} Record: {record} Total Score: {total_score} Mean Score: {mean_score:.2f} Epsilon: {agent.epsilon:.2f} Loss: 0')

    # Save the final model at the end of training
    agent.model.save(os.path.join(train_folder, 'final_model.pth'))

if __name__ == '__main__':
    os.makedirs('trains', exist_ok=True)
    train(max_games=config.MAX_GAMES)
