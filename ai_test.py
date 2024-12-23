# ai_test.py

import unittest
from unittest.mock import patch
from ai import Truck  # Import the Truck class here


class TestTruckSimulation(unittest.TestCase):
    def setUp(self):
        # Initialize a Truck object with a small grid for testing
        self.grid_width = 5
        self.grid_height = 5
        self.truck = Truck(self.grid_width, self.grid_height, object_probability=0.5, hurdle_probability=0.2, speed=1)

    # (Rest of your test code here)


class TestTruckSimulation(unittest.TestCase):

    def setUp(self):
        # Initialize a Truck object with a small grid for testing
        self.grid_width = 5
        self.grid_height = 5
        self.truck = Truck(self.grid_width, self.grid_height, object_probability=0.5, hurdle_probability=0.2, speed=1)

    def test_grid_generation(self):
        # Test that grid is generated with correct dimensions
        self.assertEqual(len(self.truck.grid), self.grid_height)
        self.assertEqual(len(self.truck.grid[0]), self.grid_width)

        # Test that the grid contains expected types (True, False, or 999 for hurdles)
        for row in self.truck.grid:
            for cell in row:
                self.assertIn(cell, [True, False, 999])

    def test_cost_generation(self):
        # Test that costs are generated with correct dimensions
        self.assertEqual(len(self.truck.costs), self.grid_height)
        self.assertEqual(len(self.truck.costs[0]), self.grid_width)

        # Test that each cell cost is either 999 (for hurdle) or a value between 1 and 10
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.truck.grid[y][x] == 999:
                    self.assertEqual(self.truck.costs[y][x], 999)
                else:
                    self.assertTrue(1 <= self.truck.costs[y][x] <= 10)

    def test_reset_position(self):
        # Test that the truck's initial position is valid and not on a hurdle
        self.truck.reset_position()
        self.assertNotEqual(self.truck.grid[self.truck.y][self.truck.x], 999)

    def test_find_nearest_object(self):
        # Test that the function correctly identifies the nearest object
        # Place an object manually and see if it is detected
        self.truck.grid[1][1] = True  # Place an object at (1, 1)
        self.truck.x, self.truck.y = 0, 0  # Set truck at (0, 0)
        nearest_object = self.truck.find_nearest_object()
        self.assertEqual(nearest_object, (1, 1))

    def test_dijkstra_pathfinding(self):
        # Test Dijkstra's algorithm to find path between two points
        start = (0, 0)
        goal = (4, 4)

        # Set a clear path in a small 5x5 grid
        self.truck.grid = [[False for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        path = self.truck.dijkstra(start, goal)

        # Verify that a path is returned and that it leads to the goal
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], goal)

    def test_collect_object(self):
        # Test that the truck collects objects properly
        self.truck.grid[2][2] = True  # Place an object at (2, 2)
        self.truck.x, self.truck.y = 2, 2  # Move truck to (2, 2)
        self.truck.collect_object()  # Collect the object

        # Verify that the object was collected
        self.assertIn((2, 2), self.truck.collected_objects)
        self.assertEqual(self.truck.grid[2][2], False)  # Object is removed from the grid

    def test_simulation_performance(self):
        # Test that the performance calculation does not crash and returns a reasonable value
        self.truck.simulate(1)  # Run one simulation
        # Performance should be calculated at the end
        # (No assert needed here since we are only checking for runtime errors)

    def test_truck_does_not_start_on_hurdle(self):
        # Ensure truck does not start on a hurdle
        with patch('random.randint') as mock_randint:
            # Force the truck to start at a specific position that’s not a hurdle
            mock_randint.side_effect = [1, 1]
            self.truck.grid[1][1] = 999  # Set (1,1) as hurdle
            self.truck.reset_position()
            self.assertNotEqual(self.truck.grid[self.truck.y][self.truck.x], 999)

    def test_no_path_due_to_hurdles(self):
        # Fill the grid with hurdles except start and end points
        start = (0, 0)
        goal = (4, 4)
        self.truck.grid = [[999 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.truck.grid[0][0] = False  # Start point
        self.truck.grid[4][4] = False  # Goal point

        path = self.truck.dijkstra(start, goal)
        # There should be no path due to hurdles blocking the way
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
