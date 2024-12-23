import matplotlib.pyplot as plt
from typing import List, Dict


def plot_training_metrics(
        rewards: List[float],
        losses: List[float],
        performance_metrics: Dict[str, List[float]]
):

    plt.ion()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(rewards, label='Episode Rewards')
    ax1.set_title('Rewards Over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()

    ax2.plot(losses, label='Training Loss', color='red')
    ax2.set_title('Loss Over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()

    success_rates = performance_metrics['success_rate']
    ax3.plot(success_rates, label='Success Rate', color='green')
    ax3.set_title('Success Rate Over Episodes')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.legend()

    plt.tight_layout()

    plt.show(block=True)