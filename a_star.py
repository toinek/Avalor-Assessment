"""Module voor het plannen van drone-paden in een grid met A*-algoritme."""

import time
import numpy as np
import sys
from heapq import heappush, heappop  # Voor prioriteitswachtrij
from greedy_rollout import Grid, load_grid_from_file  # Importeer Grid en load functie

class DroneAStar:
    """Een Drone object dat een pad plant met A*, prioriterend op hoge scores.

    Args:
        start_pos (tuple): Startpositie van de drone.
        grid (Grid): Het grid waarin de drone opereert.

    Attributes:
        position (tuple): Huidige positie.
        path (list): Bezochte posities.
        grid (Grid): Het grid.
        total_score (float): Totale verzamelde score.
    """
    def __init__(self, start_pos, grid):
        self.position = start_pos
        self.path = [start_pos]
        self.grid = grid
        self.total_score = self.grid.get_score(*start_pos)
        self.max_score = np.max(self.grid.initial_scores)  # Voor heuristiek

    def heuristic(self, pos, t_current):
        """Heuristiek: combineert afstand tot hoogste score en huidige score.

        Args:
            pos (tuple): Positie om te evalueren.
            t_current (int): Huidige tijdstap.

        Returns:
            float: Geschatte waarde van de positie.
        """
        # Eenvoudige heuristiek: huidige score + potentieel naar max score
        current_score = self.grid.get_score(*pos)
        # Afstand tot een hoge score kan complexer, hier simpel geschat
        return current_score + self.max_score * 0.5  # Gewogen toekomstige potentie

    def get_neighbors(self, pos):
        """Haal geldige buurposities op.

        Args:
            pos (tuple): Huidige positie.

        Returns:
            list: Lijst van geldige buurposities.
        """
        x, y = pos
        moves = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        new_positions = np.array([x, y]) + moves
        valid_mask = (new_positions[:, 0] >= 0) & (new_positions[:, 0] < self.grid.size) & \
                     (new_positions[:, 1] >= 0) & (new_positions[:, 1] < self.grid.size)
        return [tuple(p) for p in new_positions[valid_mask]]

    def plan_path(self, t, T):
        """Plant een pad met A* voor t stappen binnen tijdslimiet T.

        Args:
            t (int): Aantal tijdstappen.
            T (int): Tijdslimiet in milliseconden.

        Returns:
            tuple: (pad, totale score).
        """
        start_time = time.time()
        for step in range(t):
            if (time.time() - start_time) * 1000 > T:
                print(f"Gestopt na {step} stappen vanwege tijdslimiet T={T}ms")
                break

            # A* implementatie voor één stap
            open_set = []  # Prioriteitswachtrij: (f_score, g_score, positie)
            heappush(open_set, (0, 0, self.position))
            came_from = {}
            g_score = {self.position: 0}  # Totale score tot nu toe
            f_score = {self.position: self.heuristic(self.position, step)}  # g + h

            while open_set:
                _, current_g, current = heappop(open_set)

                # Beste zet gevonden voor deze stap
                neighbors = self.get_neighbors(current)
                best_next = None
                best_score = -float('inf')

                for next_pos in neighbors:
                    tentative_g = g_score[current] + self.grid.get_score(*next_pos)
                    if next_pos not in g_score or tentative_g > g_score[next_pos]:
                        came_from[next_pos] = current
                        g_score[next_pos] = tentative_g
                        h = self.heuristic(next_pos, step)
                        f = tentative_g + h
                        heappush(open_set, (f, tentative_g, next_pos))
                        if tentative_g > best_score:
                            best_score = tentative_g
                            best_next = next_pos

                if best_next:
                    break  # Stop na eerste iteratie voor één stap

            # Beweeg naar de beste positie
            if best_next and best_next != self.position:
                new_score = self.grid.get_score(*best_next)
                self.position = best_next
                self.path.append(best_next)
                self.total_score += new_score
                self.grid.visit(*best_next, step)

            # Update scores na beweging
            self.grid.update_scores()

        return self.path, self.total_score

def main(filename, start_pos, t, T):
    """Voer A*-padplanning uit en print resultaten.

    Args:
        filename (str): Bestand met grid-data.
        start_pos (tuple): Startpositie.
        t (int): Aantal tijdstappen.
        T (int): Tijdslimiet in milliseconden.
    """
    grid_data = load_grid_from_file(filename)
    grid = Grid(grid_data)
    drone = DroneAStar(start_pos, grid)
    path, score = drone.plan_path(t, T)
    print(f"Grid grootte: {grid.size}x{grid.size}")
    print(f"Pad: {path}")
    print(f"Totale score: {score}")

if __name__ == "__main__":
    filename = "1000.txt"
    start_pos = (6, 0)
    t = 50
    T = 1000
    main(filename, start_pos, t, T)
