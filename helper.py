import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, losses, epsilons, folder, game_number, config):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(f'LR: {config["LR"]}, Batch Size: {config["BATCH_SIZE"]}, Max Memory: {config["MAX_MEMORY"]}', 'Speed: {config["SPEED"]}, Max Games: {config["MAX_GAMES"]},'
            f'Reward Fruit: {config["REWARD_FRUIT"]}, Reward Collision: {config["REWARD_COLLISION"]}, Reward Step: {config["REWARD_STEP"]}')
    plt.xlabel('Number of Games')
    plt.ylabel('Metrics')

    if scores:
        plt.plot(scores, label='Score')
        plt.text(len(scores)-1, scores[-1] if scores[-1] is not None else 0, str(scores[-1] if scores[-1] is not None else 0))

    if mean_scores:
        plt.plot(mean_scores, label='Mean Score')
        plt.text(len(mean_scores)-1, mean_scores[-1] if mean_scores[-1] is not None else 0, str(mean_scores[-1] if mean_scores[-1] is not None else 0))

    if losses:
        plt.plot(losses, label='Loss')
        plt.text(len(losses)-1, losses[-1] if losses[-1] is not None else 0, str(losses[-1] if losses[-1] is not None else 0))

    if epsilons:
        plt.plot(epsilons, label='Epsilon')
        plt.text(len(epsilons)-1, epsilons[-1] if epsilons[-1] is not None else 0, str(epsilons[-1] if epsilons[-1] is not None else 0))

    plt.ylim(ymin=0)
    plt.legend()
    plt.show(block=False)
    
    # Save the plot every 30 games
    if game_number % 30 == 0:
        plt.savefig(f'{folder}/plot_{game_number}.png')
    
    plt.pause(.1)
