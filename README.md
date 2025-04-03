# Drone Padplanning Assessment

Dit project implementeert een drone-padplanningsalgoritme in een NxN grid met dynamische scores. Een implementatie voor een losse drone en een swarm drones is beschikbaar:

- **`greedy_rollout.py`**: Een greedy algoritme met optionele lookahead voor één drone. Scores groeien na bezoek in 10 stappen terug naar hun initiële waarde.
- **`swarm.py`**: Een swarm-algoritme dat plant voor meerdere drones, hergebruikmakend van de `Grid`-, en 'Drone'-klasse uit `greedy_rollout.py`.

## Gebruik
- Run `python greedy_rollout.py` of `python swarm.py` met een grid-bestand (bijv. "20.txt").
- Parameters: `start_pos`, `t` (stappen), `T` (tijdslimiet in ms), en voor greedy `lookahead`.
