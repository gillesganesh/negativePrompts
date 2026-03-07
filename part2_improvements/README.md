# Partie 2 — Axes d'amélioration

Extension de la reproduction (Partie 1) par trois axes d'amélioration choisis.

---

## Axe 1 — Few-shot (exemples en contexte)

Le papier original teste uniquement en zero-shot.
Notre amélioration : ajouter 5 démonstrations par tâche pour guider le modèle.

**Script** : `part1_reproduction/run_experiment.py --few_shot`

```python
%run /kaggle/working/negativePrompts/part1_reproduction/run_experiment.py --few_shot
```

**Résultats** : `results/protocol_vicuna_fewshot.csv`

---

## Axe 2 — Sélection automatique du stimulus optimal (Step 5)

Au lieu d'essayer tous les NP à chaque fois, sélectionner automatiquement
le meilleur stimulus par tâche à partir des résultats de l'expérience.

**Script** : `part2_improvements/stimulus_selector.py`

```python
%run /kaggle/working/negativePrompts/part2_improvements/stimulus_selector.py
```

**Sortie** : `results/stimulus_selection.csv` + rapport textuel

---

## Axe 3 — Reformulation automatique des prompts (Step 6)

Générer des variantes du prompt original combinées avec le meilleur stimulus
sélectionné automatiquement.

**Script** : `part2_improvements/prompt_reformulator.py`

```python
%run /kaggle/working/negativePrompts/part2_improvements/prompt_reformulator.py
```

**Sortie** : `results/reformulated_prompts.csv`

---

## Axe 4 — Extension multi-modèles (Step 7)

Appliquer la même méthodologie à d'autres modèles pour comparer.

**Script** : `scripts/run_all_models.py`
