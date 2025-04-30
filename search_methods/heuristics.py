from sokoban.map import Map
from typing import List, Tuple

def manhattan_distance(x1, y1, x2, y2):
    """Calculate the Manhattan distance between two points"""
    return abs(x1 - x2) + abs(y1 - y2)

def hungarian_algorithm(boxes: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> int:
    """
    Calculate the minimum cost matching between boxes and targets using Hungarian algorithm
    """
    if not boxes or not targets:
        return 0
    
    # Create cost matrix: distance from each box to each target
    cost_matrix = []
    for box in boxes:
        row = []
        for target in targets:
            row.append(manhattan_distance(box[0], box[1], target[0], target[1]))
        cost_matrix.append(row)
    
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))


def is_wall(map_state: Map, x: int, y: int) -> bool:
    """Check if position (x,y) is a wall"""
    if x < 0 or x >= map_state.length or y < 0 or y >= map_state.width:
        return True
    return map_state.map[x][y] == 1


def player_box_distance(map_state: Map) -> int:
    """
    Calculate distance from player to nearest box that needs to be moved
    """
    player_x, player_y = map_state.player.x, map_state.player.y
    targets = set(map_state.targets)
    
    # Find boxes not on targets
    movable_boxes = []
    for box_pos in map_state.positions_of_boxes.keys():
        if box_pos not in targets:
            movable_boxes.append(box_pos)
    
    if not movable_boxes:
        return 0
    
    # Find minimum distance to any box that needs to be moved
    min_dist = float('inf')
    for box_x, box_y in movable_boxes:
        dist = manhattan_distance(player_x, player_y, box_x, box_y)
        min_dist = min(min_dist, dist)
    
    return min_dist



def zone_analysis(map_state: Map) -> int:
    """
    Analyze the map for zones and check if boxes can reach their targets.
    Returns a penalty if boxes are trapped in wrong zones
    """
    penalty = 0
    boxes = set(map_state.positions_of_boxes.keys())
    targets = set(map_state.targets)
    
    boxes_not_on_targets = boxes - targets
    if not boxes_not_on_targets:
        return 0
    
    # Find all passable spaces
    passable = set()
    for x in range(map_state.length):
        for y in range(map_state.width):
            if not is_wall(map_state, x, y):
                passable.add((x, y))
    
    # Identify separate zones using flood fill
    zones = []
    remaining = passable.copy()
    
    while remaining:
        start = next(iter(remaining))
        
        zone = set()
        queue = [start]
        
        while queue:
            pos = queue.pop(0)
            if pos in zone:
                continue
                
            zone.add(pos)
            x, y = pos
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                
                if next_pos in remaining and next_pos not in zone:
                    if next_pos not in boxes:
                        queue.append(next_pos)
        
        zones.append(zone)
        
        remaining -= zone
    
    # For each zone, check if there are enough targets for the boxes
    for zone in zones:
        zone_boxes = zone & boxes
        zone_targets = zone & targets
        
        if len(zone_boxes) > len(zone_targets):
            penalty += (len(zone_boxes) - len(zone_targets)) * 5

        
        # Additional penalty for zones with boxes but no targets
        if zone_boxes and not zone_targets:
            penalty += len(zone_boxes) * 10
    
    
    return penalty


def is_pull_move(map_state, move):
    """Determine if a box move is a pull move based on player and box positions"""
    if move <= 4:
        return False
    
    implicit_move = move - 4
    player = map_state.player
    future_position = player.get_future_position(implicit_move)
    
    # If player's future position doesn't contain a box, it's a pull move
    if future_position not in map_state.positions_of_boxes:
        opposite_position = player.get_opposite_position(implicit_move)
        if opposite_position in map_state.positions_of_boxes:
            return True
    
    return False



def min_manhattan_to_target(box_position, targets):
    """Find the minimum Manhattan distance from a box to any target"""
    box_x, box_y = box_position
    return min(manhattan_distance(box_x, box_y, target_x, target_y) for target_x, target_y in targets)

def sum_of_manhattan_distances(map_state: Map):
    """
    Calculate the sum of Manhattan distances from each box to its nearest target.
    """
    total_distance = 0
    
    for box_pos in map_state.positions_of_boxes.keys():
        total_distance += min_manhattan_to_target(box_pos, map_state.targets)
    
    return total_distance



def sokoban_advanced_heuristic(map_state: Map) -> int:
    """
    Advanced heuristic for Sokoban combining multiple factors:
    - Optimal box-target matching distance
    - Player-to-box distance
    - Zone analysis
    - Pull move penalty
    """
    boxes = list(map_state.positions_of_boxes.keys())
    targets = list(map_state.targets)
    
    matching_cost = hungarian_algorithm(boxes, targets)
    
    player_distance = player_box_distance(map_state) // 2
    
    zone_penalty = zone_analysis(map_state)

    pull_move_penalty = 0
    if hasattr(map_state, 'last_move') and hasattr(map_state, 'is_last_move_pull') and map_state.is_last_move_pull:
        pull_move_penalty = 80
    
    total_cost = matching_cost + player_distance + zone_penalty + pull_move_penalty
    
    return total_cost


def sokoban_simple_heuristic(map_state: Map) -> int:
    """
    Advanced heuristic for Sokoban combining multiple factors:
    - Box-target matching distance
    - Player-to-box distance
    """


    matching_cost = sum_of_manhattan_distances(map_state) * 2

    player_distance = player_box_distance(map_state)
    
    total_cost = matching_cost + player_distance
    
    return total_cost

