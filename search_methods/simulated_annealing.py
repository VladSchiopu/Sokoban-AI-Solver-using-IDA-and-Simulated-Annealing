import random
import math
import time
from typing import List, Tuple, Optional, Callable, Set

from sokoban.map import Map
from sokoban.moves import *
from search_methods.heuristics import *


class SimulatedAnnealingSolver:
    def __init__(self, initial_map: Map, heuristic_func: Callable[[Map], float], use_pull_moves: bool = False):
        """
        Initialize the solver with the initial map state.
        
        Args:
            initial_map: The initial map state
            heuristic_func: Function to evaluate a map state
            use_pull_moves: Whether to allow pull moves (box moves)
        """
        self.initial_map = initial_map
        self.heuristic_func = heuristic_func
        self.use_pull_moves = use_pull_moves
        self.best_state = initial_map
        self.best_score = self.heuristic_func(initial_map)
        self.move_history = []
        self.best_move_history = []


        self.corner_deadlocks = self.find_corner_deadlocks(initial_map)
        self.wall_deadlocks = self.find_wall_deadlocks(initial_map)

        # Set to track visited states to avoid cycles
        self.visited_states = set()
        random.seed(0)


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
    
    def get_weighted_move(self, current_map: Map) -> Optional[int]:
        """Get a move, giving preference to moves that seem promising."""
        possible_moves = current_map.filter_possible_moves()
        
        if not self.use_pull_moves:
            possible_moves = [move for move in possible_moves if move <= DOWN]
            
        if not possible_moves:
            return None
        
        # Try each move and rate them
        move_ratings = []
        for move in possible_moves:
            new_state = current_map.copy()
            new_state.apply_move(move)
            state_hash = self.hash_state(new_state)

            new_state.last_move = move
            new_state.is_last_move_pull = is_pull_move(new_state, move)
            
            if state_hash in self.visited_states:
                continue

            if self.is_deadlock(new_state):
                continue

            self.initial_map.explored_states += 1
                
            rating = self.heuristic_func(new_state)
            # Add some randomness to avoid getting stuck
            rating += random.uniform(-0.5, 0.5)
            move_ratings.append((rating, move))
        
        # If all moves lead to previously seen states, pick randomly
        if not move_ratings:
            return random.choice(possible_moves)
            
        move_ratings.sort()
        
        # With some probability, pick the best move; otherwise, choose randomly
        if random.random() < 0.8:
            return move_ratings[0][1]
        else:
            return random.choice([m for _, m in move_ratings])
    
    def get_energy(self, map_state: Map) -> float:
        """Calculate the energy (cost) of a state."""
        if map_state.is_solved():
            return 0
        
        return self.heuristic_func(map_state)
    
    def remove_cycles_from_solution(self, move_history):
        """Remove any cycles from the solution path."""
        states = []
        state_to_index = {}
        current_state = self.initial_map.copy()
        
        state_hash = self.hash_state(current_state)
        states.append(current_state.copy())
        state_to_index[state_hash] = 0
        
        clean_moves = []
        
        for i, move in enumerate(move_history):
            current_state.apply_move(move)
            state_hash = self.hash_state(current_state)
            
            if state_hash in state_to_index:
                cycle_start = state_to_index[state_hash]
                clean_moves = clean_moves[:cycle_start]
                states = states[:cycle_start+1]
                state_to_index = {self.hash_state(s): idx for idx, s in enumerate(states)}
            else:
                clean_moves.append(move)
                states.append(current_state.copy())
                state_to_index[state_hash] = len(states) - 1
        
        return clean_moves
    
    def acceptance_probability(self, old_energy: float, new_energy: float, temperature: float) -> float:
        """
        Calculate the acceptance probability.
        """
        # If the new state is better, accept it
        if new_energy < old_energy:
            return 1.0
        
        # Otherwise, calculate acceptance probability
        return math.exp((old_energy - new_energy) / temperature)
    
    def solve(self, max_iterations: int = 50000, 
              initial_temp: float = 100.0, 
              cooling_rate: float = 0.995,
              time_limit: float = 300.0,
              restart_threshold: int = 1000) -> Optional[List[int]]:
        """
        Solve the Sokoban puzzle using Simulated Annealing.    
        Returns:
            List of moves that solve the puzzle, or None if no solution is found
        """
        start_time = time.time()
        
        current_state = self.initial_map.copy()
        current_energy = self.get_energy(current_state)
        
        best_state = current_state.copy()
        best_energy = current_energy
        
        temperature = initial_temp
        current_move_history = []
        best_move_history = []
        
        self.visited_states = set()
        self.visited_states.add(self.hash_state(current_state))
        
        iteration = 0
        moves_since_improvement = 0
        
        while iteration < max_iterations and (time.time() - start_time) < time_limit:
            if current_state.is_solved():
                self.best_state = current_state
                self.best_move_history = current_move_history.copy()
                return self.remove_cycles_from_solution(best_move_history)
            
            # Get a move that prefers good states but sometimes explores
            move = self.get_weighted_move(current_state)
            
            # If no valid moves, restart from best known state
            if move is None:
                current_state = best_state.copy()
                current_energy = best_energy
                current_move_history = best_move_history.copy()
                self.visited_states.clear()
                continue
            
            new_state = current_state.copy()
            new_state.apply_move(move)
            
            # Check if we've seen this state before
            state_hash = self.hash_state(new_state)
            if state_hash in self.visited_states:
                # With a small probability, allow revisiting states
                if random.random() > 0.05:
                    continue

            self.initial_map.explored_states += 1
            
            new_energy = self.get_energy(new_state)
            
            # Decide whether to accept the new state
            if self.acceptance_probability(current_energy, new_energy, temperature) > random.random():
                current_state = new_state
                current_energy = new_energy
                current_move_history.append(move)
                self.visited_states.add(state_hash)
                
                # Update best state if this is better
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy
                    best_move_history = current_move_history.copy()
                    moves_since_improvement = 0
                else:
                    moves_since_improvement += 1
            else:
                moves_since_improvement += 1
            
            temperature *= cooling_rate
            iteration += 1
            
            # Restart if we haven't improved for a specific number of moves
            if moves_since_improvement >= restart_threshold:
                if random.random() < 0.5:
                    current_state = best_state.copy()
                    current_energy = best_energy
                    current_move_history = best_move_history.copy()
                else:
                    current_state = self.initial_map.copy()
                    current_energy = self.get_energy(current_state)
                    current_move_history = []
                
                
                #restart_threshold += 1
                temperature = initial_temp
                moves_since_improvement = 0
                self.visited_states.clear()
                self.visited_states.add(self.hash_state(current_state))
            
        
        if best_state.is_solved():
            self.best_state = best_state
            self.best_move_history = best_move_history
            return self.remove_cycles_from_solution(best_move_history)
        
        
        return None
    
    def get_solution_moves(self) -> List[int]:
        """Get the list of moves in the solution."""
        return self.best_move_history
    
    def get_solution_length(self) -> int:
        """Get the length of the solution."""
        return len(self.best_move_history)