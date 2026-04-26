from collections import deque


def bfs_path(graph, start, goal):
    visited = set()
    queue = deque([start])
    visited.add(start)
    parent = {start: None}  # to reconstruct path

    while queue:
        current = queue.popleft()

        if current == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]  # reverse the path

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return None  # No path found

city_graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E", "G"],
    "G": ["F"]
}

start_city = "A"
end_city = "G"

path = bfs_path(city_graph, start_city, end_city)

if path:
    print(f"Path from {start_city} to {end_city}: {' -> '.join(path)}")
else:
    print(f"No path found from {start_city} to {end_city}.")