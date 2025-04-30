from sokoban.map import Map
from search_methods.ida_star import IDAStarSolver
from search_methods.simulated_annealing import SimulatedAnnealingSolver
from search_methods.heuristics import *

class Solver:
    def __init__(self, map: Map) -> None:
        self.map = map
        self.solution = None
        
    def solve(self, algorithm='ida_star', use_pull_moves=False, heuristic_func = sokoban_advanced_heuristic):
        """
        Solve the Sokoban puzzle using the specified algorithm
        """
        if algorithm == 'ida_star':
            solver = IDAStarSolver(self.map, heuristic_func, use_pull_moves)
            self.solution = solver.solve()
        elif algorithm == 'simulated_annealing':
            solver = SimulatedAnnealingSolver(self.map, heuristic_func, use_pull_moves)
            self.solution = solver.solve(
                max_iterations=200000,
                initial_temp=3000.0,
                cooling_rate=0.999,
                time_limit=300.0,
                restart_threshold=50
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented yet")
            
        return self.solution
    
    
    def visualize_solution(self, output_dir='images', base_name='solution'):
        """
        Create a gif visualizing the solution
        """
        if not self.solution:
            print("No solution to visualize")
            return
            
        from sokoban import save_images, create_gif
        
        current_map = self.map.copy()
        
        current_map.save_map(output_dir, f"{base_name}_0.png")
        images_dir = output_dir
        
        for i, move in enumerate(self.solution):
            current_map.apply_move(move)
            current_map.save_map(output_dir, f"{base_name}_{i+1}.png")
        
        create_gif(
            path_images=images_dir,
            gif_name=f"{base_name}.gif",
            save_path=output_dir
        )
        
        print(f"Solution visualization saved to {output_dir}/{base_name}.gif")