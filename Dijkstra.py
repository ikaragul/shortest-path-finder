import heapq
import time

def dijkstra_with_counters(N, S, D):
    """
    Runs Dijkstra's algorithm with repetition counters.
    N: number of nodes (1..N)
    S: source node
    D: destination node
    Returns: (shortest_path, cost_of_path, total_repetitions)
    """
    
    # ----------------------------
    # Build adjacency list
    # ----------------------------
    # adjacency[node] = list of (neighbor, weight)
    adjacency = [[] for _ in range(N+1)]
    
    # Counter for adjacency building
    adjacency_build_count = 0
    
    for i in range(1, N+1):
        for j in range(max(1, i-3), min(N, i+3)+1):
            adjacency_build_count += 1
            if j != i:
                w = i + j
                adjacency[i].append((j, w))
    
    # ----------------------------
    # Initialize shortest path variables
    # ----------------------------
    g = [float('inf')] * (N+1)  # Best-known cost from S to each node
    g[S] = 0  # Cost to reach the source is 0
    predecessor = [-1] * (N+1)  # To reconstruct the shortest path
    
    # min-heap of (g[node], node)
    min_heap = [(0, S)]
    visited = [False] * (N+1)
    
    # Counters
    extract_min_count = 0
    relaxation_count = 0
    
    # ----------------------------
    # Dijkstra's main loop
    # ----------------------------
    while min_heap:
        extract_min_count += 1
        current_g, u = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        if u == D:
            # Destination reached
            break
        
        for (v, w_uv) in adjacency[u]:
            relaxation_count += 1
            if not visited[v]:
                tentative_g = g[u] + w_uv
                if tentative_g < g[v]:
                    g[v] = tentative_g
                    predecessor[v] = u
                    heapq.heappush(min_heap, (g[v], v))
    
    if g[D] == float('inf'):
        # No path found
        return ([], float('inf'),
                adjacency_build_count + extract_min_count + relaxation_count)
    
    # ----------------------------
    # Reconstruct the path
    # ----------------------------
    path = []
    node = D
    while node != -1:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    
    total_repetitions = adjacency_build_count + extract_min_count + relaxation_count
    return (path, g[D], total_repetitions)


if __name__ == "__main__":
    N = int(input("Enter N: "))  # Number of nodes
    S = int(input("Enter S: "))  # Source node
    D = int(input("Enter D: "))  # Destination node

    start_time = time.time()
    shortest_path, cost, total_reps = dijkstra_with_counters(N, S, D)
    end_time = time.time()

    if cost == float('inf'):
        print(f"No path from {S} to {D}.")
    else:
        print(f"Shortest path from {S} to {D} is: {shortest_path}")
        print(f"Total cost of this path is: {cost}")
    print(f"Total repetition count: {total_reps}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")