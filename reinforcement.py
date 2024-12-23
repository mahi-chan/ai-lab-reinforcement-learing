import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from collections import deque
from typing import Tuple, Set, List
import os
from visualization import plot_training_metrics


class GridEnvironment:
    def __init__(self, rows: int, cols: int, object_probability: float):
        if not 0 <= object_probability <= 1:
            raise ValueError("object_probability must be between 0 and 1")
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")

        self.rows = rows
        self.cols = cols
        self.object_probability = object_probability
        self.grid = np.zeros((rows, cols))
        self.reset()

    def spawn_objects(self) -> None:
        self.grid = np.random.choice(
            [0, 1],
            size=(self.rows, self.cols),
            p=[1 - self.object_probability, self.object_probability]
        )

    def reset(self) -> np.ndarray:
        self.grid = np.zeros((self.rows, self.cols))
        self.spawn_objects()
        return self.grid.copy()

    def get_state_size(self) -> int:
        return self.rows * self.cols


class LearningRobot:
    def __init__(self, env: GridEnvironment, probabilities: np.ndarray, learning_rate: float = 0.05):
        if not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if probabilities.shape != (env.rows, env.cols):
            raise ValueError("probabilities shape must match environment dimensions")

        self.env = env
        self.probabilities = probabilities.copy()
        self.learning_rate = learning_rate
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.PROBABILITY_THRESHOLD = 0.40

        self.performance_metrics: Dict[str, List[float]] = {
            'total_cells_visited': [],
            'successful_cells': [],
            'error_cells': [],
            'success_rate': []
        }

    def get_valid_actions(self) -> List[int]:
        valid_actions = []
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                if (self.probabilities[i][j] > self.PROBABILITY_THRESHOLD and
                        (i, j) not in self.visited_cells):
                    action = i * self.env.cols + j
                    valid_actions.append(action)
        return valid_actions

    def evaluate_and_update(self, target_cell):
        x, y = target_cell
        reward = 0
        is_successful = False

        if (x, y) not in self.visited_cells:
            self.visited_cells.add((x, y))

            if self.probabilities[x][y] > self.PROBABILITY_THRESHOLD:
                if self.env.grid[x][y] == 1:
                    self.probabilities[x][y] += self.learning_rate
                    self.probabilities[x][y] = min(self.probabilities[x][y], 1.0)
                    reward = 1
                    is_successful = True
                else:
                    self.probabilities[x][y] -= self.learning_rate
                    self.probabilities[x][y] = max(self.probabilities[x][y], 0.0)
                    reward = -1

        return reward, self.probabilities, is_successful

    def record_episode_performance(self, is_successful: bool):
        total_object_cells = np.sum(self.env.grid)

        object_found_cells = sum(
            1 for cell in self.visited_cells if self.env.grid[cell[0], cell[1]] == 1
        )

        total_cells_visited = len(self.visited_cells)

        error_cells = sum(
            1 for cell in self.visited_cells if self.env.grid[cell[0], cell[1]] == 0
        )

        self.performance_metrics['total_cells_visited'].append(total_cells_visited)
        self.performance_metrics['successful_cells'].append(object_found_cells)
        self.performance_metrics['error_cells'].append(error_cells)

        success_rate = object_found_cells / total_cells_visited if total_cells_visited > 0 else 0
        self.performance_metrics['success_rate'].append(success_rate)

        print(f"Total Cells Visited: {total_cells_visited}, "
              f"Objects Found: {object_found_cells}, "
              f"Error Cells: {error_cells}, "
              f"Success Rate: {success_rate:.2%}")

        return {
            'total_cells': total_cells_visited,
            'successful_cells': object_found_cells,
            'error_cells': error_cells,
            'success_rate': success_rate
        }


class DQNAgent:
    def __init__(
            self,
            state_size: int,
            action_size: int,
            robot: LearningRobot,
            memory_size: int = 2000,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            epsilon_decay: float = 0.99,
            epsilon_min: float = 0.01,
            learning_rate: float = 0.0005
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.robot = robot
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(48, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(24, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(
            loss=Huber(delta=1),
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        valid_actions = self.robot.get_valid_actions()

        if not valid_actions:
            return -1

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        q_values = self.model.predict(state, verbose=0)[0]
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def replay(self, batch_size: int) -> float:
        if len(self.memory) < batch_size:
            return 0.0

        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0.0

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )

            target_f = self.model.predict(state, verbose=0)

            original_target = target_f[0][action]
            target_f[0][action] = target

            history = self.model.fit(
                state,
                target_f,
                epochs=1,
                verbose=0
            )

            loss = history.history['loss'][0]
            total_loss += loss

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size


def run_simulation(
        env: GridEnvironment,
        robot: LearningRobot,
        agent: DQNAgent,
        num_episodes: int,
        batch_size: int
) -> Tuple[List[float], List[float]]:
    total_rewards = []
    total_losses = []

    for episode in range(num_episodes):
        state = env.reset()
        robot.visited_cells.clear()

        episode_reward = 0
        state = state.flatten().reshape(1, -1)
        episode_successful = False
        episode_loss = 0.0

        while True:
            action = agent.act(state)

            if action == -1:
                break

            x, y = divmod(action, env.cols)
            reward, probabilities, is_successful = robot.evaluate_and_update((x, y))

            if is_successful:
                episode_successful = True

            next_state = probabilities.flatten().reshape(1, -1)

            episode_reward += reward
            agent.remember(state, action, reward, next_state, False)
            state = next_state

            valid_actions = robot.get_valid_actions()
            if not valid_actions:
                break

        if len(agent.memory) >= batch_size:
            loss = agent.replay(batch_size)
            episode_loss = loss

            performance = robot.record_episode_performance(episode_successful)

            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Loss: {loss:.4f}, "
                  )
        else:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}")

        total_rewards.append(episode_reward)
        total_losses.append(episode_loss)

    print("\nPerformance Summary:")
    print(f"Average Total Cells Visited: {np.mean(robot.performance_metrics['total_cells_visited']):.2f}")
    print(f"Average Successful Cells: {np.mean(robot.performance_metrics['successful_cells']):.2f}")
    print(f"Average Error Cells: {np.mean(robot.performance_metrics['error_cells']):.2f}")
    print(f"Overall Average Success Rate: {np.mean(robot.performance_metrics['success_rate']):.2%}")

    return total_rewards, total_losses


def main():
    ROWS, COLS = 5, 7
    OBJECT_PROBABILITY = 0.6
    NUM_EPISODES = 50
    BATCH_SIZE = 32
    SAVE_DIR = "models"

    os.makedirs(SAVE_DIR, exist_ok=True)

    env = GridEnvironment(ROWS, COLS, OBJECT_PROBABILITY)

    prob_path = os.path.join(SAVE_DIR, "environment.npy")
    try:
        probabilities = np.load(prob_path)
    except FileNotFoundError:
        probabilities = np.random.rand(ROWS, COLS)

    robot = LearningRobot(env, probabilities)
    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_state_size(),
        robot=robot
    )

    try:
        rewards, losses = run_simulation(env, robot, agent, NUM_EPISODES, BATCH_SIZE)

        np.save(prob_path, robot.probabilities)
        agent.model.save(os.path.join(SAVE_DIR, "dqn_model.h5"))
        np.save(os.path.join(SAVE_DIR, "training_rewards.npy"), rewards)
        np.save(os.path.join(SAVE_DIR, "training_losses.npy"), losses)

        plot_training_metrics(
            rewards=rewards,
            losses=losses,
            performance_metrics=robot.performance_metrics
        )

        print("Training complete. Models, data, and visualizations saved successfully.")

    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()
