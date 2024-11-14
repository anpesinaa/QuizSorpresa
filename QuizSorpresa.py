import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import random

# Parámetros del entorno y del robot
GRID_WIDTH = 20
GRID_HEIGHT = 20
NUM_OBSTACLES = 50
ROBOT_DIMENSIONS = (1, 1)  # Largo y ancho del robot
OBSTACLE_DIMENSIONS = (1, 1)  # Largo y ancho de los obstáculos
START = (0, 0)
END = (GRID_WIDTH - 1, GRID_HEIGHT - 1)

# Generar posición inicial aleatoria de los obstáculos
def create_environment():
    obstacles = set()
    while len(obstacles) < NUM_OBSTACLES:
        x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
        if (x, y) not in [START, END]:
            obstacles.add((x, y))
    return obstacles

# Algoritmo A* para encontrar el camino más corto
def a_star(start, end, obstacles):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == end:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            if neighbor in obstacles:
                continue
            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No se encontró un camino

# Heurística de Manhattan
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Reconstruir el camino
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Vecinos válidos dentro de los límites de la cuadrícula
def get_neighbors(node):
    x, y = node
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = [(x + dx, y + dy) for dx, dy in directions]
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT]

# Mover obstáculos de forma aleatoria sin salir de la cuadrícula
def move_obstacles(obstacles):
    new_obstacles = set()
    for (x, y) in obstacles:
        valid_moves = get_neighbors((x, y))
        new_position = random.choice(valid_moves)
        # Asegurar que el obstáculo no se mueva fuera de la cuadrícula ni se superponga con otros obstáculos
        if new_position not in [START, END] and new_position not in new_obstacles:
            new_obstacles.add(new_position)
        else:
            new_obstacles.add((x, y))
    return new_obstacles

# Calcular la distancia mínima al obstáculo más cercano
def min_distance_to_obstacle(position, obstacles):
    x, y = position
    return min(abs(x - ox) + abs(y - oy) for ox, oy in obstacles)

# Calcular la distancia al objetivo
def distance_to_goal(position, goal):
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

# Configuración de la animación
fig, ax = plt.subplots()
path_line, = ax.plot([], [], "b-", linewidth=2, label="Path")
start_point = ax.plot(START[1], START[0], "go", markersize=10, label="Start")[0]
end_point = ax.plot(END[1], END[0], "ro", markersize=10, label="End")[0]
obstacle_points = ax.plot([], [], "ks", markersize=10, label="Obstacles")[0]
robot_dot, = ax.plot([], [], "bo", markersize=8, label="Robot")

ax.set_xlim(-1, GRID_WIDTH)
ax.set_ylim(-1, GRID_HEIGHT)
ax.invert_yaxis()
plt.legend()

# Inicializar posición de obstáculos y camino
obstacles = create_environment()
path = a_star(START, END, obstacles)

# Listas para almacenar la distancia al objetivo y la distancia al obstáculo más cercano en cada paso
distances_to_goal = []
distances_to_nearest_obstacle = []

# Función de actualización de animación
def update(frame):
    global obstacles, path
    
    # Mover obstáculos aleatoriamente
    obstacles = move_obstacles(obstacles)
    # Recalcular el camino
    path = a_star(START, END, obstacles)

    # Actualizar posición de obstáculos en la visualización
    ox, oy = zip(*obstacles) if obstacles else ([], [])
    obstacle_points.set_data(oy, ox)

    # Actualizar el camino en la visualización y calcular las distancias
    if path:
        px, py = zip(*path)
        path_line.set_data(py, px)
        # Mover el robot a lo largo del camino
        if frame < len(path):
            current_position = path[frame]
            robot_dot.set_data([py[frame]], [px[frame]])
            # Calcular la distancia al objetivo y la distancia al obstáculo más cercano
            distance_to_goal_current = distance_to_goal(current_position, END)
            distance_to_obstacle = min_distance_to_obstacle(current_position, obstacles)
            distances_to_goal.append(distance_to_goal_current)
            distances_to_nearest_obstacle.append(distance_to_obstacle)
        else:
            robot_dot.set_data([], [])
    else:
        path_line.set_data([], [])  # Borrar si no hay camino
        robot_dot.set_data([], [])

    return path_line, obstacle_points, robot_dot

# Crear y ejecutar la animación
ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=True)
plt.title("Simulación del Robot con Obstáculos Móviles")
plt.show()

# Graficar los resultados de la distancia al objetivo
plt.figure()
plt.plot(distances_to_goal, marker='o')
plt.xlabel('Paso')
plt.ylabel('Distancia al objetivo')
plt.title('Distancia del robot al objetivo durante el recorrido')
plt.show()

# Graficar los resultados de la distancia al obstáculo más cercano
plt.figure()
plt.plot(distances_to_nearest_obstacle, marker='o')
plt.xlabel('Paso')
plt.ylabel('Distancia al obstáculo más cercano')
plt.title('Distancia del robot al obstáculo más cercano durante el recorrido')
plt.show()
