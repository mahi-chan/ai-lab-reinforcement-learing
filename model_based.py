import random
import pygame
import sys

class Truck:
    def __init__(self, grid_width, grid_height, object_probability=0.5, hurdle_probability=0.125, speed=3):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.object_probability = object_probability
        self.hurdle_probability = hurdle_probability
        self.speed = speed
        self.grid = self.generate_grid()
        self.reset_position()

    def generate_grid(self):
        grid = []
        for _ in range(self.grid_height):
            row = []
            for _ in range(self.grid_width):
                if random.random() < self.hurdle_probability:
                    row.append(999)
                else:
                    row.append(random.random() < self.object_probability)
            grid.append(row)
        return grid

    def reset_position(self):
        while True:
            self.x = random.randint(1, self.grid_width - 2)
            self.y = random.randint(1, self.grid_height - 2)
            if self.grid[self.y][self.x] != 999:
                break
        self.locations = [(self.x, self.y)]
        self.collected_objects = []
        self.last_position = None

    def move(self):
        if self.x == 0 or self.x == self.grid_width - 1 or self.y == 0 or self.y == self.grid_height - 1:
            return False

        directions = [('up', self.x, self.y + 1), ('down', self.x, self.y - 1), ('left', self.x - 1, self.y), ('right', self.x + 1, self.y)]
        random.shuffle(directions)

        for direction, new_x, new_y in directions:
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and self.grid[new_y][new_x] == True:
                self.last_position = (self.x, self.y)
                self.x, self.y = new_x, new_y
                self.locations.append((self.x, self.y))
                self.collect_object()
                return True

        random.shuffle(directions)
        for direction, new_x, new_y in directions:
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and (new_x, new_y) != self.last_position and (new_x, new_y) not in self.locations and self.grid[new_y][new_x] != 999:
                self.last_position = (self.x, self.y)
                self.x, self.y = new_x, new_y
                self.locations.append((self.x, self.y))
                self.collect_object()
                return True

        return False

    def collect_object(self):
        if self.grid[self.y][self.x] == True:
            self.grid[self.y][self.x] = False
            self.collected_objects.append((self.x, self.y))

    def draw_grid(self, screen, cell_size):
        colors = {
            999: (0, 255, 255),
            True: (255, 255, 0),
            False: (255, 255, 255)
        }
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                color = colors[self.grid[y][x]]
                pygame.draw.rect(screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size), 1)

        for x, y in self.collected_objects:
            pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
        for i, (x, y) in enumerate(self.locations):
            alpha = 255 - int(255 * (i / len(self.locations)))
            trail_color = (255, 0, 0, alpha)
            pygame.draw.rect(screen, trail_color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))

    def simulate(self, simulations, cell_size=60):
        pygame.init()
        screen = pygame.display.set_mode((self.grid_width * cell_size, self.grid_height * cell_size))
        clock = pygame.time.Clock()

        total_movements = 0
        total_collected_objects = 0

        for _ in range(simulations):
            self.reset_position()
            print("Initial grid:")
            self.print_grid()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                if not self.move():
                    running = False

                screen.fill((0, 0, 0))
                self.draw_grid(screen, cell_size)
                pygame.display.flip()
                clock.tick(self.speed)

            total_movements += len(self.locations)
            total_collected_objects += len(self.collected_objects)

            print("Truck locations:", self.locations)
            print("Collected object positions:", self.collected_objects)
            print("Collected grid positions:")
            self.print_collected_grid()
            print()

        performance = total_movements / total_collected_objects if total_collected_objects > 0 else 0
        print(f"Performance: {performance:.2f}")

    def print_grid(self):
        for row in self.grid:
            print(' '.join(['O' if cell == True else 'H' if cell == 999 else '.' for cell in row]))

    def print_collected_grid(self):
        object_grid = [['O' if cell == True else 'H' if cell == 999 else '.' for cell in row] for row in self.grid]
        for x, y in self.collected_objects:
            object_grid[y][x] = 'C'
        for row in object_grid:
            print(' '.join(row))

if __name__ == "__main__":
    grid_width = 8
    grid_height = 9
    truck = Truck(grid_width, grid_height, speed=3)
    truck.simulate(5)
