from model_based import Truck

def test_generate_grid():
    grid_width, grid_height = 5, 5
    truck = Truck(grid_width, grid_height)

    assert len(truck.grid) == grid_height
    assert all(len(row) == grid_width for row in truck.grid)

    valid_values = {True, False, 999}
    for row in truck.grid:
        for cell in row:
            assert cell in valid_values

def test_reset_position():
    grid_width, grid_height = 5, 5
    truck = Truck(grid_width, grid_height)

    truck.reset_position()
    x, y = truck.x, truck.y

    assert 0 <= x < grid_width
    assert 0 <= y < grid_height

    assert truck.grid[y][x] != 999

def test_move():
    grid_width, grid_height = 5, 5
    truck = Truck(grid_width, grid_height)

    truck.grid = [
        [False, False, False, False, False],
        [False, 999, True, False, False],
        [False, False, False, False, False],
        [False, False, 999, False, False],
        [False, False, False, False, False],
    ]
    truck.x, truck.y = 2, 2

    moved = truck.move()

    assert moved is True
    assert truck.grid[truck.y][truck.x] == False

def test_collect_object():
    grid_width, grid_height = 5, 5
    truck = Truck(grid_width, grid_height)

    truck.grid[2][2] = True
    truck.x, truck.y = 2, 2

    truck.collect_object()

    assert truck.grid[2][2] == False
    assert (2, 2) in truck.collected_objects

def test_print_grid(capsys):
    grid_width, grid_height = 5, 5
    truck = Truck(grid_width, grid_height)

    truck.grid = [
        [False, False, 999, False, True],
        [True, False, False, 999, False],
        [False, False, False, True, False],
        [999, False, False, False, False],
        [False, True, False, 999, False],
    ]

    truck.print_grid()
    captured = capsys.readouterr()

    expected_output = (
        ". . H . O\n"
        "O . . H .\n"
        ". . . O .\n"
        "H . . . .\n"
        ". O . H .\n"
    )

    assert captured.out == expected_output
