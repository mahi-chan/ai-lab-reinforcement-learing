import unittest
from utility_based import Truck
import pygame

class TestTruck(unittest.TestCase):

    def setUp(self):

        self.truck = Truck(grid_width=8, grid_height=9, speed=2)

    def test_initialization(self):

        self.assertEqual(self.truck.grid_width, 8)
        self.assertEqual(self.truck.grid_height, 9)
        self.assertEqual(self.truck.speed, 2)
        self.assertIsInstance(self.truck.grid, list)
        self.assertIsInstance(self.truck.costs, list)

    def test_generate_grid(self):

        grid = self.truck.generate_grid()
        self.assertEqual(len(grid), 9)
        self.assertEqual(len(grid[0]), 8)
        self.assertTrue(any(cell == True for row in grid for cell in row))

    def test_generate_costs(self):

        costs = self.truck.generate_costs()
        self.assertEqual(len(costs), 9)
        self.assertEqual(len(costs[0]), 8)
        self.assertTrue(all(isinstance(cell, int) for row in costs for cell in row))

    def test_reset_position(self):

        self.truck.reset_position()
        self.assertTrue(1 <= self.truck.x < self.truck.grid_width - 1)
        self.assertTrue(1 <= self.truck.y < self.truck.grid_height - 1)
        self.assertIn((self.truck.x, self.truck.y), self.truck.locations)

    def test_move(self):

        self.truck.reset_position()
        initial_position = (self.truck.x, self.truck.y)
        self.truck.move()
        new_position = (self.truck.x, self.truck.y)
        self.assertNotEqual(initial_position, new_position)

    def test_collect_object(self):

        self.truck.grid[self.truck.y][self.truck.x] = True
        self.truck.collect_object()
        self.assertIn((self.truck.x, self.truck.y), self.truck.collected_objects)
        self.assertFalse(self.truck.grid[self.truck.y][self.truck.x])

    def test_find_nearest_object(self):

        self.truck.grid[2][2] = True
        self.truck.grid[4][4] = True
        self.truck.x, self.truck.y = 3, 3
        nearest_object = self.truck.find_nearest_object()
        self.assertEqual(nearest_object, (2, 2))

    def test_draw_grid(self):

        pygame.init()
        screen = pygame.display.set_mode((self.truck.grid_width * 60, self.truck.grid_height * 60))
        self.truck.draw_grid(screen, 60)


    def test_simulate(self):

        self.truck.simulate(1)



if __name__ == '__main__':
    unittest.main()
