import random


def move_right(position, num_tiles):
    if position < num_tiles - 1:
        position += 1
    return position


def move_left(position):
    if position > 0:
        position -= 1
    return position


def suck(tiles, position):
    cleaned = 0
    if tiles[position] == 1:
        tiles[position] = 0
        cleaned = 1
    return cleaned


def run_vacuum(num_tiles):
    tiles = [random.choice([0, 1]) for _ in range(num_tiles)]
    initial_dirt_locations = sum(tiles)
    position = 0
    cleaned_tiles = 0
    turns = 0
    visited = [False] * num_tiles

    while not all(visited):
        if tiles[position] == 1:
            cleaned_tiles += suck(tiles, position)
            turns += 1
        visited[position] = True

        if position < num_tiles - 1 and not visited[position + 1]:
            position = move_right(position, num_tiles)
            turns += 1
        elif position > 0 and not visited[position - 1]:
            position = move_left(position)
            turns += 1
        else:
            break

    performance = (turns / 3) * 100
    return performance, turns, initial_dirt_locations


def simulate(num_tiles, iterations):
    total_performance = 0
    total_turns = 0
    total_dirt_locations = 0

    for i in range(iterations):
        performance, turns, initial_dirt_locations = run_vacuum(num_tiles)
        total_performance += performance
        total_turns += turns
        total_dirt_locations += initial_dirt_locations

    avg_performance = total_turns / total_dirt_locations
    print(f"\nAverage Performance after {iterations} iterations: {avg_performance}%")
    print(f"Total Actions Taken: {total_turns}")
    print(f"Total Dirt Locations: {total_dirt_locations}")


if __name__ == "__main__":
    num_tiles = 2
    iterations = 1000
    simulate(num_tiles, iterations)
