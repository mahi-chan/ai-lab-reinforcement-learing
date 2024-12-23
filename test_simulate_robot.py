import numpy as np
from simulate_robot import GridEnvironment, LearningRobot, run_simulation

def test_grid_environment_initialization():
    rows, cols, object_probability = 5, 7, 0.2
    env = GridEnvironment(rows, cols, object_probability)

    assert env.rows == rows
    assert env.cols == cols
    assert env.object_probability == object_probability
    assert env.grid.shape == (rows, cols)

def test_grid_environment_object_spawning():
    rows, cols, object_probability = 5, 7, 0.5
    env = GridEnvironment(rows, cols, object_probability)
    env.spawn_objects()

    assert np.all((env.grid == 0) | (env.grid == 1))
    assert np.any(env.grid == 1)

def test_grid_environment_reset():
    rows, cols, object_probability = 5, 7, 0.5
    env = GridEnvironment(rows, cols, object_probability)
    env.spawn_objects()

    env.reset()
    assert np.all(env.grid == 0)

def test_robot_navigation():
    rows, cols = 5, 7
    probabilities = np.random.rand(rows, cols)
    env = GridEnvironment(rows, cols, 0.2)
    robot = LearningRobot(env, probabilities)

    path = robot.navigate()

    path_probabilities = [probabilities[cell[0], cell[1]] for cell in path]
    assert path_probabilities == sorted(path_probabilities, reverse=True)

def test_robot_evaluation():
    rows, cols = 5, 7
    probabilities = np.random.rand(rows, cols)
    env = GridEnvironment(rows, cols, 0.2)
    env.spawn_objects()
    robot = LearningRobot(env, probabilities)

    target_cells = robot.navigate()

    results, evaluation_grid = robot.evaluate(target_cells)

    assert "Success" in results
    assert "Error" in results

    assert np.all(np.isin(evaluation_grid, [-1, 0, 1]))

def test_run_simulation(tmp_path):
    rows, cols, object_probability = 5, 7, 0.2

    probabilities = np.random.rand(rows, cols)
    model_file = tmp_path / "probabilities.npy"
    np.save(model_file, probabilities)

    results, probabilities_out, evaluation_grid, path, object_grid = run_simulation(
        rows, cols, object_probability, model_file
    )

    assert results["Success"] >= 0
    assert results["Error"] >= 0
    assert probabilities_out.shape == (rows, cols)
    assert evaluation_grid.shape == (rows, cols)
    assert len(path) > 0
    assert object_grid.shape == (rows, cols)
