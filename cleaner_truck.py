import random

class Truck:
    def __init__(self, grid_width, grid_height, object_probability=0.5):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.object_probability = object_probability
        self.grid = self.generate_grid()
        self.reset_position()

    def generate_grid(self):
        return [[random.random() < self.object_probability for _ in range(self.grid_width)] for _ in range(self.grid_height)]

    def reset_position(self):
        self.x = random.randint(1, self.grid_width - 2)
        self.y = random.randint(1, self.grid_height - 2)
        self.locations = [(self.x, self.y)]
        self.collected_objects = []

    def move(self):
        if self.x == 0 or self.x == self.grid_width - 1 or self.y == 0 or self.y == self.grid_height - 1:
            return False
        possible_directions = ['up', 'down', 'left', 'right']
        random.shuffle(possible_directions)
        for direction in possible_directions:
            new_x, new_y = self.x, self.y
            if direction == 'up' and self.y < self.grid_height - 1:
                new_y += 1
            elif direction == 'down' and self.y > 0:
                new_y -= 1
            elif direction == 'left' and self.x > 0:
                new_x -= 1
            elif direction == 'right' and self.x < self.grid_width - 1:
                new_x += 1
            if (new_x, new_y) not in self.locations:
                self.x, self.y = new_x, new_y
                self.locations.append((self.x, self.y))
                self.collect_object()
                return True
        return False

    def collect_object(self):
        if self.grid[self.y][self.x]:
            self.grid[self.y][self.x] = False
            self.collected_objects.append((self.x, self.y))

    def print_grid(self):
        for row in self.grid:
            print(' '.join(['O' if cell else '.' for cell in row]))

    def print_collected_grid(self):
        object_grid = [['O' if cell else '.' for cell in row] for row in self.grid]
        for x, y in self.collected_objects:
            object_grid[y][x] = 'C'
        for row in object_grid:
            print(' '.join(row))

    def simulate(self, simulations):
        total_locations = set()
        total_collected_objects = []

        for _ in range(simulations):
            self.reset_position()
            print("initial grid:")
            self.print_grid()
            while self.move():
                pass
            total_locations.update(self.locations)
            total_collected_objects.extend(self.collected_objects)
            print("Truck locations:", self.locations)
            print("Collected object positions:", self.collected_objects)
            print("Collected grid positions:")
            self.print_collected_grid()
            print()

        total_object = len(total_collected_objects)
        performance = (len(total_locations) + total_object) / total_object \
            if total_object > 0 else 0
        print(f"Performance: {performance:.2f}")

if __name__ == "__main__":
    grid_width = 5
    grid_height = 7
    truck = Truck(grid_width, grid_height)
    truck.simulate(5)
