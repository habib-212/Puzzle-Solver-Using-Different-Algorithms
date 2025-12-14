import time
import heapq
from collections import deque

def display_8puzzle(puzzle):
    """
    Displays the 8-puzzle grid in a 3x3 format.
    """
    print("\nCurrent 8-Puzzle State:")
    for row in puzzle:
        print(" ".join(str(num) if num != 0 else " " for num in row))
    print()

#----------------- 8-Puzzle Solver ----------------

# Defining possible moves for the blank tile
MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1)
}

class EightPuzzle:
  def __init__(self, initial_state, goal_state):
    self.initial_state = initial_state
    self.goal_state = goal_state

  def get_blank_position(self, state):
    #Return the (row, col) position of the blank tile
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

  def generate_neighbors(self, state):
    """
    Generate all possible moves (neighbors) from the current state.
    Each neighbor is a tuple (new_state, move_name).
    """
    neighbors = []
    x, y = self.get_blank_position(state)
    for move, (dx, dy) in MOVES.items():
      new_x, new_y = x + dx, y + dy
      # Check if move is within bounds
      if 0 <= new_x < 3 and 0 <= new_y < 3:
        new_state = [row[:] for row in state]   # Creates a deep copy of the state to swap tiles
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        neighbors.append((new_state, move))
    return neighbors

def bfs_solver(puzzle):

  #Returns a tuple: (solution path as a list of moves, nodes expanded, time taken)

  start_time = time.time()
  initial = puzzle.initial_state
  goal = puzzle.goal_state

  # Each element in the queue is a tuple: (current_state, path_so_far)
  queue = deque([(initial, [])])
  visited = set()
  nodes_expanded = 0

  while queue:
    state, path = queue.popleft()
    nodes_expanded += 1

    if state == goal:
      time_taken = time.time() - start_time
      return path, nodes_expanded, time_taken   # When solution found

    # Convert state to a tuple of tuples (hashable) for the visited set
    state_tuple = tuple(tuple(row) for row in state)
    if state_tuple in visited:
      continue
    visited.add(state_tuple)

    # Generate new states from the current state
    for neighbor, move in puzzle.generate_neighbors(state):
      queue.append((neighbor, path + [move]))

  time_taken = time.time() - start_time
  return None, nodes_expanded, time_taken  # If no solution found

def dfs_solver(puzzle, max_depth=100):

  #Returns a tuple: (solution path, nodes expanded, time taken)

  start_time = time.time()
  initial = puzzle.initial_state
  goal = puzzle.goal_state

  # Stack for DFS: each element is (state, path_so_far)
  stack = [(initial, [])]
  visited = set()
  nodes_expanded = 0

  while stack:
    state, path = stack.pop()
    nodes_expanded += 1

    if state == goal:
      time_taken = time.time() - start_time
      return path, nodes_expanded, time_taken

    # Limit search depth to avoid infinite loops in DFS
    if len(path) >= max_depth:
      continue      # Breakpoint

    state_tuple = tuple(tuple(row) for row in state)
    if state_tuple in visited:
      continue
    visited.add(state_tuple)

    for neighbor, move in puzzle.generate_neighbors(state):
      stack.append((neighbor, path + [move]))

  time_taken = time.time() - start_time
  return None, nodes_expanded, time_taken

def manhattan_distance(state, goal):
  """
  Calculate the Manhattan Distance heuristic.
  For each tile (ignoring the blank), it adds the distance from its current
  position to its position in the goal state.
  """
  distance = 0
  for i in range(3):
    for j in range(3):
      value = state[i][j]
      if value != 0:
      # Find the position of the current tile in the goal state
        for x in range(3):
          for y in range(3):
            if goal[x][y] == value:
              distance += abs(x - i) + abs(y - j)
              break
  return distance

def a_star_solver(puzzle):
  """
  A* search algorithm uses Manhattan Distance as the heuristic.
  Returns a tuple: (solution path, nodes expanded, time taken)
  """
  start_time = time.time()
  initial = puzzle.initial_state
  goal = puzzle.goal_state

  # Priority queue elements: (f, g, current_state, path_so_far)
  # where f = g + h and g is the cost so far.
  open_list = []
  heapq.heappush(open_list, (manhattan_distance(initial, goal), 0, initial, []))
  visited = set()
  nodes_expanded = 0

  while open_list:
    f, g, state, path = heapq.heappop(open_list)
    nodes_expanded += 1

    if state == goal:
      return path, nodes_expanded, time.time() - start_time

    state_tuple = tuple(tuple(row) for row in state)
    if state_tuple in visited:
      continue
    visited.add(state_tuple)

    for neighbor, move in puzzle.generate_neighbors(state):
      new_g = g + 1
      h = manhattan_distance(neighbor, goal)
      heapq.heappush(open_list, (new_g + h, new_g, neighbor, path + [move]))

  time_taken = time.time() - start_time
  return None, nodes_expanded, time_taken

def get_8puzzle_input():
  try:
    user_input = input("Enter the initial 8-puzzle state (9 numbers separated by spaces, use 0 for blank): ")
    numbers = list(map(int, user_input.split()))
    if set(numbers) != set(range(9)) or len(numbers) != 9:
      raise ValueError("Invalid input. Please enter numbers 0 through 8 exactly once.")
    puzzle = [numbers[i*3:(i+1)*3] for i in range(3)]
    return puzzle
  except Exception as e:
    print("Error:", e)
    return None

# ------------------- Sudoku Solver -------------------

# Basic Sudoku Solver (Backtracking)

def find_empty(board):
  for i in range(9):
    for j in range(9):
      if board[i][j] == 0:
        return (i, j)
  return None

def is_valid(board, num, position):
  row, col = position
  # Check row
  if num in board[row]:
    return False
  # Check column
  if num in [board[i][col] for i in range(9)]:
    return False
  # Check 3x3 subgrid
  start_row = (row // 3) * 3
  start_col = (col // 3) * 3
  for i in range(3):
    for j in range(3):
      if board[start_row+i][start_col+j] == num:
        return False
  return True

def sudoku_solver_basic(board):
  empty = find_empty(board)
  if not empty:
    return True  # Puzzle solved
  row, col = empty
  for num in range(1, 10):
    if is_valid(board, num, (row, col)):
      board[row][col] = num
      if sudoku_solver_basic(board):
        return True
      board[row][col] = 0  # Backtrack
  return False

# Advanced Sudoku Solver using MRV (Minimum Remaining Value)
def find_empty_mrv(board):
  min_count = 10
  best = None
  for i in range(9):
    for j in range(9):
      if board[i][j] == 0:
        candidates = [num for num in range(1, 10) if is_valid(board, num, (i, j))]
        if len(candidates) < min_count:
          min_count = len(candidates)
          best = (i, j)
        if min_count == 1:
          return best
  return best

def sudoku_solver_advanced(board):
  empty = find_empty_mrv(board)
  if not empty:
    return True
  row, col = empty
  for num in range(1, 10):
    if is_valid(board, num, (row, col)):
      board[row][col] = num
      if sudoku_solver_advanced(board):
        return True
      board[row][col] = 0
  return False

def get_sudoku_input():
  print("Enter the Sudoku puzzle row by row.")
  print("Use 0 for empty cells. Enter 9 numbers separated by spaces per row.")
  board = []
  for i in range(9):
    try:
      row_input = input(f"Row {i+1}: ")
      row = list(map(int, row_input.split()))
      if len(row) != 9:
        raise ValueError("Each row must contain exactly 9 numbers.")
      board.append(row)
    except Exception as e:
      print("Error:", e)
      return None
  return board

def print_sudoku(board):
  for i in range(9):
    if i % 3 == 0 and i != 0:
      print("-" * 21)
    for j in range(9):
      if j % 3 == 0 and j != 0:
        print("|", end=" ")
      print(board[i][j] if board[i][j] != 0 else ".", end=" ")
    print()


#------------------ Main Program ------------------

def main():
  print("Puzzle Solver Menu: ")
  print("1. 8-Puzzle Solver")
  print("2. Sudoku Solver")

  choice = input("Enter your choice (1/2): ").strip()

  if choice == '1':
    initial = get_8puzzle_input()
    if not initial:
      return

    display_8puzzle(initial)
    goal = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]]

    puzzle = EightPuzzle(initial, goal)

    # Menu
    print("Select search algorithm for the 8-Puzzle Solver:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. A* Search")
    algorithm_choice = input("Enter your choice (1/2/3): ").strip()

    if algorithm_choice == '1':
        solution, nodes, time_taken = bfs_solver(puzzle)
        algorithm = "BFS"
    elif algorithm_choice == '2':
        solution, nodes, time_taken = dfs_solver(puzzle)
        algorithm = "DFS"
    elif algorithm_choice == '3':
        solution, nodes, time_taken = a_star_solver(puzzle)
        algorithm = "A* Search"
    else:
        print("Invalid Selection. Exiting.")
        return

    # Print results
    if solution is not None:
        print(f"\nAlgorithm: {algorithm}")
        print(f"Solution found in {len(solution)} moves: {solution}")
    else:
      
       print("No solution found.")

    print(f"Nodes expanded: {nodes}")
    print(f"Time taken: {time_taken:.4f} seconds")

  elif choice == '2':
    board = get_sudoku_input()
    if not board:
      return
    print("\nInitial Sudoku Board:")
    print_sudoku(board)

    print("\nSelect an algorithm to solve the Sudoku board:")
    print("1. Basic Backtracking")
    print("2. Advanced Backtracking (MRV heuristic)")

    sudoku_algorithm_choice = input("Enter your choice (1/2): ").strip()
    if sudoku_algorithm_choice == '1':
      start_time = time.time()
      solved = sudoku_solver_basic(board)
      algorithm_name ="Basic Backtracking"
    elif sudoku_algorithm_choice == '2':
      start_time = time.time()
      solved = sudoku_solver_advanced(board)
      algorithm_name = "Advanced Backtracking (MRV)"
    else:
      print("Invalid Selection. Exiting.")
      return

    time_taken = time.time() - start_time

    if solved:
      print(f"\nSudoku Solved using {algorithm_name}: ")
      print_sudoku(board)
      print(f"Solved in {time_taken: .4f} seconds.")
    else:
      print("No solution exists for the provided Sudoku puzzle.")
  else:
    print("Invalid selection .")

if __name__ == "__main__":
  main()
