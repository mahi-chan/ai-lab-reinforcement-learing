import random
import pygame
import sys
import heapq

class Truck:
    def __init__(self, grid_width, grid_height, object_probability=0.125, hurdle_probability=0.125, speed=3):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.object_probability = object_probability
        self.hurdle_probability = hurdle_probability
        self.speed = speed
        self.grid = self.generate_grid()
        self.costs = self.generate_costs()
        self.reset_position()
        pygame.init()
        self.screen = pygame.display.set_mode((grid_width * 60, grid_height * 60))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

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

    def generate_costs(self):
        costs = []
        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                if self.grid[y][x] == 999:  # Hurdle
                    row.append(999)
                else:
                    row.append(random.randint(1, 10))
            costs.append(row)
        return costs

    def reset_position(self):
        while True:
            self.x = random.randint(1, self.grid_width - 2)
            self.y = random.randint(1, self.grid_height - 2)
            if self.grid[self.y][self.x] != 999:
                break
        self.locations = [(self.x, self.y)]
        self.collected_objects = []
        self.collect_object()

    def dijkstra(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            current_cost, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < self.grid_width and 0 <= neighbor[1] < self.grid_height:
                    if self.grid[neighbor[1]][neighbor[0]] == 999:
                        continue

                    tentative_g_score = g_score[current] + self.costs[neighbor[1]][neighbor[0]]
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        heapq.heappush(open_set, (tentative_g_score, neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def move(self):
        target = self.find_nearest_object()
        if not target:
            return False

        path = self.dijkstra((self.x, self.y), target)
        if path and len(path) > 1:
            for next_step in path[1:]:
                self.x, self.y = next_step
                self.locations.append(next_step)
                self.collect_object()
                self.draw_grid(self.screen, 60)
                pygame.display.flip()
                self.clock.tick(self.speed)
                if self.x == 0 or self.x == self.grid_width - 1 or self.y == 0 or self.y == self.grid_height - 1:
                    return False
            return True
        else:
            return False

    def collect_object(self):
        if self.grid[self.y][self.x] == True:
            self.grid[self.y][self.x] = False
            self.collected_objects.append((self.x, self.y))

    def find_nearest_object(self):
        nearest_object = None
        min_distance = float('inf')
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y][x] == True:
                    distance = abs(self.x - x) + abs(self.y - y)  # Manhattan distance
                    if distance < min_distance:
                        min_distance = distance
                        nearest_object = (x, y)

        if nearest_object:
            path = self.dijkstra((self.x, self.y), nearest_object)
            if path:
                return nearest_object
            else:
                return None
        else:
            return None

    def draw_grid(self, screen, cell_size):
        # Define new colors for a cleaner and more modern look
        background_color = (30, 30, 30)      # Dark gray background
        obstacle_color = (220, 20, 60)       # Crimson red for obstacles
        object_color = (255, 215, 0)         # Gold for collectible objects
        truck_color = (50, 205, 50)          # Lime green for truck
        path_color = (70, 130, 180)          # Steel blue for path history
        collected_color = (100, 149, 237)    # Cornflower blue for collected objects

        screen.fill(background_color)  # Set background color

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Determine the color of each cell
                if self.grid[y][x] == 999:  # Obstacle
                    color = obstacle_color
                    pygame.draw.circle(screen, color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 3)
                elif self.grid[y][x] == True:  # Object to collect
                    color = object_color
                    pygame.draw.circle(screen, color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 3)
                else:
                    # Draw a subtle background for cells without objects
                    color = background_color
                    pygame.draw.rect(screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))

        # Draw the truck path
        for i, (x, y) in enumerate(self.locations):
            if i == len(self.locations) - 1:
                # Draw the truck itself with a distinct color
                pygame.draw.circle(screen, truck_color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 2)
            else:
                # Draw previous locations with path color
                pygame.draw.circle(screen, path_color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 4)

        # Draw collected objects with a different color
        for x, y in self.collected_objects:
            pygame.draw.circle(screen, collected_color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 3)

    def simulate(self, simulations, cell_size=60):
        pygame.init()
        screen = pygame.display.set_mode((self.grid_width * cell_size, self.grid_height * cell_size))
        clock = pygame.time.Clock()

        total_movements = 0
        total_collected_objects = 0

        for _ in range(simulations):
            self.reset_position()
            running = True
            movements_this_simulation = 0
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
                movements_this_simulation += 1

            collected_objects_this_simulation = len(self.collected_objects)
            total_movements += movements_this_simulation
            total_collected_objects += collected_objects_this_simulation

            pygame.time.delay(1000)

        performance = total_movements / total_collected_objects if total_collected_objects > 0 else 0
        print(f"Overall Performance: {performance:.2f}")

        pygame.quit()

if __name__ == "__main__":
    grid_width = 8
    grid_height = 9
    truck = Truck(grid_width, grid_height, speed=2)
    truck.simulate(4)
