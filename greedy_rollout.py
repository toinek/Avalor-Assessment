"""Module voor het plannen van drone-paden in een grid met dynamische scores."""
import time
import numpy as np
import sys
def load_grid_from_file(filename):
    """Laad een grid van scores uit een bestand.

    Args:
        filename (str): De naam van het bestand.

    Returns:
        np.ndarray: Een 2D numpy array met scores voor elk vakje in het grid.
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            grid = [[int(num) for num in line.split()] for line in lines if line.strip()]
            return np.array(grid)
    except Exception as e:
        print(f"An unexpected error occurred while loading grid: {e}")
        sys.exit(1)

class Grid:
    """Een Grid object houdt de scores bij van elk vakje in het grid en houdt bij welke vakjes bezocht zijn.

    Args:
        initial_scores (np.ndarray): Een 2D numpy array met scores voor elk vakje in het grid.

    Attributes:
        initial_scores (np.ndarray): De initiële scores van elk vakje in het grid.
        current_scores (np.ndarray): De huidige scores van elk vakje in het grid.
        visited (np.ndarray): Een boolean array die bijhoudt welke vakjes bezocht zijn.
        last_visit (np.ndarray): Een array die bijhoudt wanneer elk vakje voor het laatst bezocht is.
        size (int): De grootte van het grid (aantal vakjes in één dimensie).
    """
    def __init__(self, initial_scores):
        self.initial_scores = initial_scores.astype(float)
        self.current_scores = self.initial_scores.copy()
        self.visited = np.zeros_like(self.initial_scores, dtype=bool)
        self.last_visit = np.zeros_like(self.initial_scores, dtype=float)
        self.size = self.initial_scores.shape[0]

    def update_scores(self, growth_rate=0.1):
        """Update scores van bezochte vakken; deze groeien in 1/growth_rate stappen terug naar de initiële waarde.

        Args:
            growth_rate (float): De groeisnelheid van de scores van bezochte vakken.
        """
        self.current_scores[self.visited] = np.minimum(self.initial_scores[self.visited], self.current_scores[self.visited] + self.initial_scores[self.visited] * growth_rate)

    def get_score(self, x, y):
        """Haal de huidige score op van een vakje.

        Args:
            x: X-coördinaat.
            y: Y-coördinaat.

        Returns:
            De score op positie (x, y).
        """
        return self.current_scores[x, y]

    def visit(self, x, y, t_current):
        """Bezoek een vakje, reset de score naar 0 en markeer als bezocht.

        Args:
            x: X-coördinaat.
            y: Y-coördinaat.
            t_current: Huidige tijdstap.
        """
        self.current_scores[x, y] = 0
        self.last_visit[x, y] = t_current
        self.visited[x, y] = True

class Drone:
    """ Een Drone object houdt de positie en het pad bij van een drone, en kan een pad plannen door een grid.

    Args:
        start_pos (tuple): De startpositie van de drone.
        grid (Grid): Het grid waar de drone zich in bevindt.

    Attributes:
        position (tuple): De huidige positie van de drone.
        path (list): Een lijst van posities die de drone heeft bezocht.
        grid (Grid): Het grid waar de drone zich in bevindt.
        total_score (float): De totale score van de drone.
    """
    def __init__(self, start_pos, grid):
        self.position = start_pos
        self.path = [start_pos]
        self.grid = grid
        self.total_score = self.grid.get_score(*start_pos)

    def evaluate_path(self, start_pos, depth, t_current):
        """"Evalueert recursief de beste score van een pad vanaf een startpositie.

        Args:
            start_pos (tuple): De startpositie van het pad.
            depth (int): De diepte van de evaluatie.
            t_current (int): Huidige tijdstap.

        Returns:
            De totale score van het pad.
        """

        # Base case: stoppen als de diepte 0 is
        if depth <= 0:
            return 0
        x, y = start_pos
        moves = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        new_positions = np.array([x, y]) + moves
        valid_mask = (new_positions[:, 0] >= 0) & (new_positions[:, 0] < self.grid.size) & \
                     (new_positions[:, 1] >= 0) & (new_positions[:, 1] < self.grid.size)
        valid_positions = new_positions[valid_mask]

        # Stop als er geen geldige posities zijn
        if len(valid_positions) == 0:
            return 0

        # Recursief de beste score van de volgende stappen evalueren
        best_future_score = 0
        for next_pos in valid_positions:
            score = self.grid.current_scores[next_pos[0], next_pos[1]]
            future = self.evaluate_path(tuple(next_pos), depth - 1, t_current + 1)
            best_future_score = max(best_future_score, score + future)
        return best_future_score

    def get_next_move(self, lookahead, t_current):
        """"Bepaalt de volgende zet van de drone door lookahead stappen vooruit te kijken.

        Args:
            lookahead (int): Het aantal stappen vooruitkijken.
            t_current (int): Huidige tijdstap.

        Returns:
            De volgende positie van de drone.
        """
        x, y = self.position

        # Bepaal de mogelijke zetten
        moves = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        new_positions = np.array([x, y]) + moves
        valid_mask = (new_positions[:, 0] >= 0) & (new_positions[:, 0] < self.grid.size) & \
                     (new_positions[:, 1] >= 0) & (new_positions[:, 1] < self.grid.size)
        valid_positions = new_positions[valid_mask]
        if len(valid_positions) == 0:
            return self.position

        # Als er zetten mogelijk zijn, bepaal de score van elke mogelijke zet
        path_scores = []
        for pos in valid_positions:
            current_score = self.grid.current_scores[pos[0], pos[1]]
            future_score = self.evaluate_path(tuple(pos), lookahead - 1, t_current + 1)
            path_scores.append(current_score + future_score)
        best_idx = np.argmax(path_scores)
        return tuple(valid_positions[best_idx])

    def move_to(self, new_pos, t_current):
        """"Verplaatst de drone naar een nieuwe positie en update de score en het pad.

        Args:
            new_pos (tuple): De nieuwe positie van de drone.
            t_current (int): Huidige tijdstap.
        """
        new_score = self.grid.get_score(*new_pos)
        self.position = new_pos
        self.path.append(new_pos)
        self.total_score += new_score
        self.grid.visit(*self.position, t_current)

    def plan_path(self, t, T, lookahead=5):
        """"Plant een pad voor de drone voor t stappen, met een tijdslimiet van T milliseconden.

        Args:
            t (int): Het aantal stappen dat de drone moet plannen.
            T (int): De tijdslimiet in milliseconden.
            lookahead (int): Het aantal stappen vooruitkijken.

        Returns:
            path (list): Een lijst van posities die de drone heeft bezocht.
            total_score (float): De totale score van de drone.
        """
        start_time = time.time()
        for step in range(t):
            if (time.time() - start_time) * 1000 > T:
                print(f"Gestopt na {step} stappen vanwege tijdslimiet T={T}ms")
                break
            next_pos = self.get_next_move(lookahead, step)
            self.move_to(next_pos, step)
            self.grid.update_scores()
        return self.path, self.total_score

def main(filename, start_pos, t, T, lookahead):
    grid_data = load_grid_from_file(filename)
    grid = Grid(grid_data)
    drone = Drone(start_pos=start_pos, grid=grid)
    path, score = drone.plan_path(t, T, lookahead)
    print(f"Grid grootte: {grid.size}x{grid.size}")
    print(f"Pad: {path}")
    print(f"Totale score: {score}")

if __name__ == "__main__":
    filename = "1000.txt"
    start_pos = (6, 0)
    t = 50
    T = 1000

    for i in range(1, 2):
        lookahead = i  # Aanpasbare parameter t (lookahead)
        main(filename, start_pos, t, T, lookahead)