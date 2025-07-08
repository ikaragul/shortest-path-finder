import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Dijkstra import dijkstra_with_counters
from A_Star import a_star_with_counters

sns.set_theme()  # Applies seaborn's default theme

class AlgorithmAnalyzer:
    def __init__(self):
        self.N_values = [10, 50, 100, 200, 500, 1000, 2000]
        self.results = {
            'dijkstra': {'times': [], 'reps': [], 'paths': [], 'costs': []},
            'astar': {'times': [], 'reps': [], 'paths': [], 'costs': []}
        }
        
    def run_analysis(self):
        for N in self.N_values:
            print(f"\nAnalyzing for N = {N}")
            
            # Dijkstra's Algorithm
            start_time = time.time()
            path_d, cost_d, reps_d = dijkstra_with_counters(N, 1, N)
            time_d = time.time() - start_time
            
            self.results['dijkstra']['times'].append(time_d)
            self.results['dijkstra']['reps'].append(reps_d)
            self.results['dijkstra']['paths'].append(path_d)
            self.results['dijkstra']['costs'].append(cost_d)
            
            # A* Algorithm
            start_time = time.time()
            path_a, cost_a, reps_a = a_star_with_counters(N, 1, N)
            time_a = time.time() - start_time
            
            self.results['astar']['times'].append(time_a)
            self.results['astar']['reps'].append(reps_a)
            self.results['astar']['paths'].append(path_a)
            self.results['astar']['costs'].append(cost_a)
            
            # Print detailed results for each N
            self._print_comparison(N, time_d, time_a, reps_d, reps_a, cost_d, cost_a)
    
    def _print_comparison(self, N, time_d, time_a, reps_d, reps_a, cost_d, cost_a):
        print(f"\nN = {N}:")
        print(f"Dijkstra's Algorithm:")
        print(f"  - Time: {time_d:.6f} seconds")
        print(f"  - Repetitions: {reps_d}")
        print(f"  - Path cost: {cost_d}")
        print(f"\nA* Algorithm:")
        print(f"  - Time: {time_a:.6f} seconds")
        print(f"  - Repetitions: {reps_a}")
        print(f"  - Path cost: {cost_a}")
        print(f"\nComparison:")
        print(f"  - Time difference: {((time_d - time_a)/time_d)*100:.2f}% ({('A* faster' if time_a < time_d else 'Dijkstra faster')})")
        print(f"  - Repetition difference: {((reps_d - reps_a)/reps_d)*100:.2f}% ({('A* more efficient' if reps_a < reps_d else 'Dijkstra more efficient')})")
    
    def plot_results(self):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Execution Time
        plt.subplot(2, 2, 1)
        plt.plot(self.N_values, self.results['dijkstra']['times'], 'b-', label='Dijkstra', marker='o')
        plt.plot(self.N_values, self.results['astar']['times'], 'r-', label='A*', marker='s')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Number of Repetitions
        plt.subplot(2, 2, 2)
        plt.plot(self.N_values, self.results['dijkstra']['reps'], 'b-', label='Dijkstra', marker='o')
        plt.plot(self.N_values, self.results['astar']['reps'], 'r-', label='A*', marker='s')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Number of Repetitions')
        plt.title('Algorithm Repetitions Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Theoretical vs Actual (Dijkstra)
        plt.subplot(2, 2, 3)
        theoretical_complexity = [N * np.log(N) for N in self.N_values]
        normalized_theoretical = [t / max(theoretical_complexity) * max(self.results['dijkstra']['reps']) 
                                for t in theoretical_complexity]
        plt.plot(self.N_values, normalized_theoretical, 'b--', label='Theoretical O(N log N)')
        plt.plot(self.N_values, self.results['dijkstra']['reps'], 'b-', label='Actual Dijkstra')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Complexity')
        plt.title('Theoretical vs Actual Complexity (Dijkstra)')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Path Costs Comparison
        plt.subplot(2, 2, 4)
        plt.plot(self.N_values, self.results['dijkstra']['costs'], 'b-', label='Dijkstra', marker='o')
        plt.plot(self.N_values, self.results['astar']['costs'], 'r-', label='A*', marker='s')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Path Cost')
        plt.title('Path Cost Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        report = """
        Comparative Analysis Report: Dijkstra's Algorithm vs A* Search

        1. Time Complexity Analysis:
        --------------------------
        """
        # Add theoretical complexity
        report += "Theoretical Complexity:\n"
        report += "- Dijkstra's Algorithm: O(N log N) with min-heap implementation\n"
        report += "- A* Search: O(N log N) in worst case, but typically better in practice due to heuristic\n\n"
        
        # Add empirical results
        report += "Empirical Results:\n"
        for i, N in enumerate(self.N_values):
            report += f"\nN = {N}:\n"
            report += f"- Dijkstra: {self.results['dijkstra']['reps'][i]} repetitions, {self.results['dijkstra']['times'][i]:.6f} seconds\n"
            report += f"- A*: {self.results['astar']['reps'][i]} repetitions, {self.results['astar']['times'][i]:.6f} seconds\n"
        
        # Add advantages and disadvantages
        report += """
        2. Algorithm Comparison:
        ----------------------
        Dijkstra's Algorithm:
        Advantages:
        - Guaranteed to find the shortest path
        - No need for heuristic function
        - More predictable performance
        Disadvantages:
        - Explores more nodes than necessary
        - Generally slower than A* for targeted search

        A* Search:
        Advantages:
        - More efficient for targeted search
        - Explores fewer nodes when good heuristic is available
        - Generally faster than Dijkstra's for specific source-destination pairs
        Disadvantages:
        - Requires a good heuristic function
        - May not be optimal if heuristic is not admissible
        - Additional memory overhead for heuristic calculations

        3. Space Complexity:
        ------------------
        Both algorithms use:
        - Adjacency List: O(N + E) where E is number of edges
        - Priority Queue: O(N)
        - Distance/Cost Arrays: O(N)
        - Predecessor Array: O(N)

        Total Space Complexity: O(N + E) for both algorithms
        """
        return report

if __name__ == "__main__":
    analyzer = AlgorithmAnalyzer()
    analyzer.run_analysis()
    analyzer.plot_results()
    print(analyzer.generate_report())