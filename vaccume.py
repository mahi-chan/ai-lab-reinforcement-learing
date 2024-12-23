import random


class VacuumCleaner:
    def __init__(self, num_tiles):
        self.num_tiles = num_tiles
        self.tiles = [random.choice([0, 1]) for _ in range(num_tiles)]
        self.position = 0
        self.cleaned_tiles = 0
        self.turns = 0
        self.visited = [False] * num_tiles
        self.numberOfActions = 0
        self.numberOfDirt = 0
        self.i = 0

    def move_right(self):
        if self.position < self.num_tiles - 1:
            self.position += 1
        self.turns += 1

    def move_left(self):
        if self.position > 0:
            self.position -= 1
        self.turns += 1

    def suck(self):
        if self.tiles[self.position] == 1:
            self.tiles[self.position] = 0
            self.cleaned_tiles += 1
        self.turns += 1

    def run(self):
        while not all(self.visited) and self.i < 1000:
            self.i += 1
            if self.tiles[self.position] == 1:
                self.suck()
            self.visited[self.position] = True

            if self.position < self.num_tiles - 1 and not self.visited[self.position + 1]:
                self.move_right()
            elif self.position > 0 and not self.visited[self.position - 1]:
                self.move_left()
            else:
                break

        performance = (self.cleaned_tiles / self.num_tiles) * 100
        print(f"Performance: {performance}%")
        print(f"Turns taken: {self.turns}")


if __name__ == "__main__":
    num_tiles = 2
    vacuum = VacuumCleaner(num_tiles)
    vacuum.run()
