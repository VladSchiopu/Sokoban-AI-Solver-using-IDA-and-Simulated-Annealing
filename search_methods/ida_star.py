from sokoban.map import Map
from typing import List, Tuple, Optional, Set
from search_methods.heuristics import *
import time

class IDAStarSolver:
    def __init__(self, initial_map: Map, heuristic_func=None, use_pull_moves=False):
        """
        Initialize the IDA* solver for Sokoban
        
        Args:
            initial_map: Initial state of the map
            heuristic_func: Function to estimate distance to goal
            use_pull_moves: Whether to allow pull moves (BOX_LEFT, BOX_RIGHT, etc.)
        """
        self.initial_map = initial_map
        self.heuristic = heuristic_func if heuristic_func else self._default_heuristic
        self.use_pull_moves = use_pull_moves
        self.explored_states = 0
        self.pruned_states = 0
        self.max_depth_reached = 0
        self.start_time = None
        
        self.corner_deadlocks = self.find_corner_deadlocks(initial_map)
        self.wall_deadlocks = self.find_wall_deadlocks(initial_map)
        
        # Transposition table to track visited states
        self.transposition_table = {}
    
    
    def find_corner_deadlocks(self, map_state: Map) -> Set[Tuple[int, int]]:
        """Find positions where boxes would be stuck in corners"""
        corners = set()
        targets = set(map_state.targets)
        
        for x in range(map_state.length):
            for y in range(map_state.width):
                if (x, y) in targets or is_wall(map_state,x,y):
                    continue
                
                wall_horizontal = False
                wall_vertical = False
                
                # Check left/right walls
                if ((x > 0 and map_state.map[x-1][y] == 1) or 
                    (x == 0) or 
                    (x >= map_state.length-1)
                    or (x < map_state.length - 1 and map_state.map[x+1][y] == 1)):
                    wall_horizontal = True
                    
                # Check up/down walls
                if ((y > 0 and map_state.map[x][y-1] == 1) or 
                    (y == 0) or 
                    (y >= map_state.width-1)
                    or (y < map_state.width - 1 and map_state.map[x][y+1] == 1)):
                    wall_vertical = True
                    
                if wall_horizontal and wall_vertical:
                    corners.add((x, y))
                    
        return corners
    

    def find_wall_deadlocks(self, map_state: Map) -> Set[Tuple[int, int]]:
        """Find positions where boxes would be stuck along walls with no target on same axis (used when pull moves are not allowed)"""
        deadlocks = set()
        targets = set(map_state.targets)

        rows_with_targets = set(x for x, y in targets)
        cols_with_targets = set(y for x, y in targets)

        for x in range(map_state.length):
            for y in range(map_state.width):
                if is_wall(map_state, x, y) or (x, y) in targets:
                    continue

                wall_nearby = False
                no_axis_targets = False
                # Wall on left or right
                if x == 0 or x == map_state.length - 1:
                    wall_nearby = True
                    if x not in rows_with_targets:
                        no_axis_targets = True
                # Wall on top or bottom
                if y == 0 or y == map_state.width - 1:
                    wall_nearby = True
                    if y not in cols_with_targets:
                        no_axis_targets = True

                if wall_nearby and no_axis_targets:
                    deadlocks.add((x, y))

        return deadlocks

    
    def is_deadlock(self, map_state: Map) -> bool:
        """Check if the current state has any deadlocks"""
        targets = set(map_state.targets)
        
        for box_pos in map_state.positions_of_boxes.keys():
            if box_pos in targets:
                continue
                
            if box_pos in self.corner_deadlocks:
                return True
            
            if not self.use_pull_moves and box_pos in self.wall_deadlocks:
                return True
                
        return False

    def hash_state(self, map_state: Map) -> str:
        """Create a hash for the current map state"""
        player_pos = (map_state.player.x, map_state.player.y)
        boxes = frozenset(map_state.positions_of_boxes.keys())
        return hash((player_pos, boxes))
    

    
    def search(self, current_map: Map, g: int, bound: int, path: List[int]) -> float:
        """
        Recursive depth-first search with bound for IDA*
        Returns:
            Solution path or next bound
        """
        self.explored_states += 1
        self.max_depth_reached = max(self.max_depth_reached, g)
        
        if self.start_time and time.time() - self.start_time > 300:  # 5 minute limit
            return float('inf')
        
        if current_map.is_solved():
            return path.copy()
        
        h = self.heuristic(current_map)
        f = g + h
        
        if self.is_deadlock(current_map):
            self.pruned_states += 1
            return float('inf')
        
        if f > bound:
            return f
        
        state_hash = self.hash_state(current_map)
        
        # If we've seen this state with a better or equal path, skip it
        if state_hash in self.transposition_table and self.transposition_table[state_hash] <= g:
            self.pruned_states += 1
            return float('inf')
        
        self.transposition_table[state_hash] = g
        
        # Get and prioritize possible moves
        possible_moves = []
        for move in current_map.filter_possible_moves():
            if not self.use_pull_moves and move > 4:
                continue
            
            # Make a temporary move to evaluate
            temp_map = current_map.copy()
            temp_map.apply_move(move)
            
            if self.hash_state(temp_map) == state_hash:
                continue

            self.initial_map.explored_states+=1
            
            # Determine if this is a pull move and store that information
            temp_map.last_move = move
            temp_map.is_last_move_pull = is_pull_move(temp_map, move)
            
            move_h = self.heuristic(temp_map)
            
            if self.is_deadlock(temp_map):
                continue
                
            possible_moves.append((move_h, move))
        
        # Sort moves by heuristic value (best first)
        possible_moves.sort()
        
        # Try all moves in priority order
        min_cost = float('inf')
        for _, move in possible_moves:
            next_map = current_map.copy()
            next_map.apply_move(move)

            self.initial_map.explored_states+=1
            
            next_map.last_move = move
            next_map.is_last_move_pull = is_pull_move(next_map, move)
            
            # Recursively search from the new state
            path.append(move)
            result = self.search(next_map, g + 1, bound, path)
            
            if isinstance(result, list):
                return result
            
            # Update minimum cost
            if isinstance(result, (int, float)):
                min_cost = min(min_cost, result)
            
            path.pop()
        
        return min_cost
    
    def solve(self, time_limit=300) -> Optional[List[int]]:
        """
        Solve the Sokoban puzzle using IDA* algorithm
        
        Returns:
            List of moves if solution found, None otherwise
        """
        self.start_time = time.time()
        self.explored_states = 0
        self.pruned_states = 0
        self.max_depth_reached = 0
        
        # Initial bound is the heuristic estimate from the start
        bound = self.heuristic(self.initial_map)
        path = []
        
        print(f"Starting IDA* with initial bound: {bound}")
        
        # Iterative deepening loop
        while bound < float('inf'):
            self.transposition_table = {}
            
            if time.time() - self.start_time > time_limit:
                print(f"Time limit of {time_limit} seconds exceeded")
                return None
            
            result = self.search(self.initial_map, 0, bound, path)
            
            if isinstance(result, list):
                elapsed = time.time() - self.start_time
                print(f"Solution found! Explored: {self.explored_states} states")
                print(f"Pruned: {self.pruned_states} states")
                print(f"Max depth: {self.max_depth_reached}")
                print(f"solution length: {len(result)}")
                print(f"Time: {elapsed:.2f} seconds")
                return result
            
            # No solution possible
            if result == float('inf'):
                print(f"No solution found. Explored: {self.explored_states} states")
                return None
            
            bound = result
            print(f"Increasing bound to {bound}, explored: {self.explored_states} states")
            
        
        return None
    