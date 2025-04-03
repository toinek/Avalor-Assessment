"""" Swarm class met greedy rollout algorithm """
import time
from greedy_rollout import Grid, Drone, load_grid_from_file  # Importeer klassen uit je originele bestand

class Swarm:
    """Een Swarm object houdt een groep drones bij en plant hun paden door een grid.

    Args:
        start_positions (list): Een lijst van startposities voor de drones.
        grid (Grid): Het grid waar de drones zich in bevinden.

    Attributes:
        drones (list): Een lijst van Drone objecten.
        grid (Grid): Het grid waar de drones zich in bevinden.
        num_drones (int): Het aantal drones in de swarm.
    """
    def __init__(self, start_positions, grid):
        self.drones = [Drone(pos, grid) for pos in start_positions]
        self.grid = grid
        self.num_drones = len(start_positions)

    def get_next_moves(self, lookahead, t_current):
        """"Bepaalt de volgende zetten van de drones door lookahead stappen vooruit te kijken.

        Args:
            lookahead (int): Het aantal stappen vooruitkijken.
            t_current (int): Huidige tijdstap.

        Returns:
            Een lijst van nieuwe posities voor de drones
        """
        # Haal voorgestelde zetten op voor alle drones
        proposed_moves = [drone.get_next_move(lookahead, t_current) for drone in self.drones]
        # Controleer op conflicten (meerdere drones naar hetzelfde vak)
        move_counts = {}
        for i, move in enumerate(proposed_moves):
            if move in move_counts:
                move_counts[move].append(i)
            else:
                move_counts[move] = [i]

        # Los conflicten op: alleen de eerste drone krijgt voorrang
        final_moves = proposed_moves.copy()
        for move, drone_indices in move_counts.items():
            if len(drone_indices) > 1:  # Conflict
                # Alle drones na de eerste krijgen hun huidige positie (geen beweging)
                for idx in drone_indices[1:]:
                    final_moves[idx] = self.drones[idx].position

        return final_moves

    def plan_swarm_path(self, t, T, lookahead=5):
        """"Plant de paden van de drones door het grid.

        Args:
            t (int): Het aantal stappen dat de drones moeten plannen.
            T (int): De tijdslimiet in milliseconden.
            lookahead (int): Het aantal stappen vooruitkijken.

        Returns:
            paths (list): Een lijst van paden die de drones hebben bezocht.
            total_score (float): De totale score van alle drones.
        """
        start_time = time.time()
        total_score = sum(drone.total_score for drone in self.drones)  # Startscore van alle drones

        for step in range(t):
            if (time.time() - start_time) * 1000 > T:
                print(f"Gestopt na {step} stappen vanwege tijdslimiet T={T}ms")
                break

            # Update scores in het grid
            self.grid.update_scores()

            # Bepaal volgende zetten voor alle drones
            next_moves = self.get_next_moves(lookahead, step)

            # Voer bewegingen uit en verzamel scores
            visited_this_step = set()  # Houd bij welke vakken deze stap bezocht zijn
            for i, (drone, new_pos) in enumerate(zip(self.drones, next_moves)):
                if new_pos == drone.position:
                    continue  # Geen beweging, geen score-update

                new_score = self.grid.get_score(*new_pos)
                if new_pos in visited_this_step:
                    new_score = 0  # Geen score als het vak al bezocht is deze stap
                else:
                    visited_this_step.add(new_pos)

                drone.position = new_pos
                drone.path.append(new_pos)
                drone.total_score += new_score
                self.grid.visit(*new_pos, step)
                total_score += new_score

        # Verzamel paden en totale score
        paths = [drone.path for drone in self.drones]
        return paths, total_score


def main(filename, start_positions, t, T, lookahead):
    grid_data = load_grid_from_file(filename)
    grid = Grid(grid_data)
    swarm = Swarm(start_positions, grid)
    paths, total_score = swarm.plan_swarm_path(t, T, lookahead)
    print(f"Grid grootte: {grid.size}x{grid.size}")
    print(f"Aantal drones: {len(start_positions)}")
    for i, path in enumerate(paths):
        print(f"Pad drone {i + 1}: {path}")
    print(f"Totale score: {total_score}")


if __name__ == "__main__":
    filename = "100.txt"
    start_positions = [(0, 0), (10, 19), (0, 19), (19, 0), (19, 19), (9, 9), (5, 5), (15, 15), (5, 15), (15, 5)]
    t = 50
    T = 1000
    lookahead = 3
    main(filename, start_positions, t, T, lookahead)