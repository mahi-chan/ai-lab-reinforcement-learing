import pytest
import numpy as np
from reinforcement import GridEnvironment, LearningRobot


def test_grid_environment_initialization():
    env = GridEnvironment(rows=5, cols=7, object_probability=0.6)

    assert env.rows == 5
    assert env.cols == 7
    assert env.object_probability == 0.6

    initial_state = env.reset()
    assert isinstance(initial_state, np.ndarray)
    assert initial_state.shape == (5, 7)


def test_grid_environment_invalid_parameters():
    with pytest.raises(ValueError, match="object_probability must be between 0 and 1"):
        GridEnvironment(rows=5, cols=7, object_probability=1.5)

    with pytest.raises(ValueError, match="rows and cols must be positive integers"):
        GridEnvironment(rows=-1, cols=7, object_probability=0.5)


def test_learning_robot_valid_actions():
    env = GridEnvironment(rows=3, cols=3, object_probability=0.5)
    probabilities = np.full((3, 3), 0.5)
    robot = LearningRobot(env, probabilities, learning_rate=0.05)

    valid_actions = robot.get_valid_actions()
    assert len(valid_actions) == 9

    robot.visited_cells = {(0, 0), (1, 1)}
    valid_actions = robot.get_valid_actions()

    assert len(valid_actions) == 7
    assert all((i, j) not in robot.visited_cells for i, j in [divmod(a, 3) for a in valid_actions])


def test_learning_robot_evaluate_and_update():
    env = GridEnvironment(rows=3, cols=3, object_probability=1.0)
    env.grid[1, 1] = 1  # Manually place an object
    probabilities = np.full((3, 3), 0.5)
    robot = LearningRobot(env, probabilities, learning_rate=0.05)

    reward, updated_probs, is_successful = robot.evaluate_and_update((1, 1))

    assert reward == 1
    assert is_successful == True
    assert updated_probs[1, 1] > 0.5
    assert (1, 1) in robot.visited_cells

    reward, updated_probs, is_successful = robot.evaluate_and_update((1, 1))

    assert reward == 0
    assert is_successful == False
    assert updated_probs[1, 1] > 0.5


