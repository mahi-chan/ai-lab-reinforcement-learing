import unittest
from vaccume1000 import move_right, move_left, suck, simulate, run_vacuum

class TestVacuum(unittest.TestCase):

    def test_move_right(self):
        self.assertEqual(move_right(0, 2), 1)
        print("move_right passed")

    def test_move_left(self):
        self.assertEqual(move_left(1), 0)
        print("move_left passed")

    def test_suck(self):
        tiles = [0, 1, 0]
        self.assertEqual(suck(tiles, 1), 1)
        print("suck passed")

    def test_simulate(self):
        simulate(2, 10)
        print("simulate passed")

if __name__ == "__main__":
    unittest.main()
