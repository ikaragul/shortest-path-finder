import heapq
import time

def a_star_with_counters(N, S, D):
    """
    Runs A* search with repetition counters.
    N: number of nodes (1..N)
    S: source node
    D: destination node
    Returns: (shortest_path, cost_of_path, total_repetitions)
    """
    
    # ----------------------------
    # Build adjacency list
    # ----------------------------
    adjacency = [[] for _ in range(N+1)]
    adjacency_build_count = 0
    
    for i in range(1, N+1):
        for j in range(max(1, i-3), min(N, i+3)+1):
            adjacency_build_count += 1
            if j != i:
                w = i + j
                adjacency[i].append((j, w))
    
    # ----------------------------
    # Initialize g, f
    # g[v] = best-known cost from S to v
    # f[v] = g[v] + h(v) = priority for the min-heap
    # ----------------------------
    g = [float('inf')] * (N+1)
    f = [float('inf')] * (N+1)
    predecessor = [-1] * (N+1)
    
    # A simple heuristic h(i) = abs(D - i)
    def h(i):
        return abs(D - i)
    
    g[S] = 0
    f[S] = h(S)
    
    # min-heap of (f[node], node)
    min_heap = [(f[S], S)]
    
    visited = [False]*(N+1)
    
    # Counters
    extract_min_count = 0
    relaxation_count = 0
    decrease_key_count = 0
    
    while min_heap:
        extract_min_count += 1
        current_f, u = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        if u == D:
            # reconstruct path
            break
        
        for (v, w_uv) in adjacency[u]:
            relaxation_count += 1
            if not visited[v]:
                tentative_g = g[u] + w_uv
                if tentative_g < g[v]:
                    g[v] = tentative_g
                    f[v] = g[v] + h(v)
                    predecessor[v] = u
                    decrease_key_count += 1
                    heapq.heappush(min_heap, (f[v], v))
    
    if g[D] == float('inf'):
        # no path
        return ([], float('inf'),
                adjacency_build_count + extract_min_count + relaxation_count + decrease_key_count)
    
    # Reconstruct the path
    path = []
    node = D
    while node != -1:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    
    total_repetitions = adjacency_build_count + extract_min_count + relaxation_count + decrease_key_count
    return (path, g[D], total_repetitions)


if __name__ == "__main__":
    N = int(input("Enter N: "))  # Number of nodes
    S = int(input("Enter S: "))  # Source node
    D = int(input("Enter D: "))  # Destination node

    start_time = time.time()
    shortest_path, cost, total_reps = a_star_with_counters(N, S, D)
    end_time = time.time()

    if cost == float('inf'):
        print(f"No path from {S} to {D}.")
    else:
        print(f"Shortest path from {S} to {D} is: {shortest_path}")
        print(f"Total cost is: {cost}")
    print(f"Total repetition count: {total_reps}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")