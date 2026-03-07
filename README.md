# NegativePrompt — Reproduction & Extension (PPD M2)

Reproduction et extension de [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024).

---

## Objectifs du projet

1. **Reproduction minimale** — 5 tâches représentatives, Vicuna-13B, prompt original vs NP01–NP10
2. **Protocole clair** — tracking automatique (modèle, prompt, paramètres, métrique, score)
3. **Analyse des stimuli** — effet de chaque stimulus par tâche, clustering
4. **Sélection automatique du stimulus** — modèle de recommandation du meilleur NP par tâche
5. **Reformulation des prompts** — amélioration automatique du prompt original
6. **Extension multi-modèles** — T5, Llama-2, Vicuna (si temps disponible)
7. **Industrialisation** — pipeline end-to-end (si temps disponible)

---

## Statut actuel

- [x] Dépôt fonctionnel, erreurs d'import corrigées
- [x] Modèles papier intégrés (T5, Vicuna-13B, Llama-2-13B via HuggingFace)
- [x] Script optimisé Kaggle (modèle chargé une seule fois)
- [x] Reproduction minimale : 5 tâches × Vicuna × NP00–NP10 avec protocole CSV
- [ ] Analyse des résultats et clustering
- [ ] Modèle de sélection automatique du stimulus
- [ ] Reformulation des prompts

---

## Tâches sélectionnées (reproduction minimale)

| Tâche | Type | Métrique |
|-------|------|---------|
| `sentiment` | Classification binaire | EM sentiment |
| `antonyms` | Génération lexicale | EM contain |
| `translation_en-fr` | Traduction | EM contain |
| `cause_and_effect` | Raisonnement causal | EM causal |
| `larger_animal` | Connaissance factuelle | EM animal |

---

## Modèles et GPU requis

| Modèle | HuggingFace | fp16 | 4-bit (P100/T4) |
|--------|-------------|------|-----------------|
| T5 | `google/flan-t5-large` | ~3 GB | — |
| Vicuna | `lmsys/vicuna-13b-v1.5` | ~26 GB (T4×2) | ~7 GB (P100/T4) |
| Llama-2 | `meta-llama/Llama-2-13b-chat-hf` | ~26 GB (T4×2) | ~7 GB (P100/T4) |

> **P100 (16 GB) et T4 (16 GB)** : utiliser le flag `--quantize` (4-bit bitsandbytes).
> **T4×2 (32 GB)** : fp16 natif, pas de quantification nécessaire.

---

## Lancer sur Kaggle Notebook (P100 ou T4×2)

### Cellule 1 — Setup (une seule fois par session)

```python
import subprocess, os

REPO = "/kaggle/working/negativePrompts"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", "branche_chen",
                    "https://github.com/agitfirat/negativePrompts", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "origin", "branche_chen"], check=True)

subprocess.run(["pip", "install", "-r", f"{REPO}/requirements.txt", "-q"], check=True)
print("Setup OK")
```

### Cellule 2 — Reproduction minimale Vicuna (5 tâches)

```python
# P100 (16 GB) — quantification 4-bit obligatoire
%run /kaggle/working/negativePrompts/run_vicuna_5tasks.py --quantize

# T4×2 (32 GB) — fp16 natif
# %run /kaggle/working/negativePrompts/run_vicuna_5tasks.py
```

### Cellule 3 — Voir les résultats

```python
import pandas as pd
df = pd.read_csv("/kaggle/working/negativePrompts/results/protocol_vicuna.csv")
print(df.to_string())
```

---

## Scripts disponibles

| Script | Description |
|--------|-------------|
| `run_vicuna_5tasks.py` | **Reproduction minimale** — Vicuna, 5 tâches, NP00–NP10, protocole CSV |
| `run_all_kaggle.py` | Toutes les tâches (II + BigBench), tous les modèles |
| `main.py` | Instruction Induction, une tâche à la fois (CLI) |
| `main_bigbench.py` | BigBench, une tâche à la fois (CLI) |

---

## Toutes les tâches disponibles (Instruction Induction)

```
active_to_passive, antonyms, cause_and_effect, common_concept, diff,
first_word_letter, informal_to_formal, larger_animal, letters_list,
negation, num_to_verbal, orthography_starts_with, rhymes,
second_word_letter, sentence_similarity, sentiment, singular_to_plural,
sum, synonyms, taxonomy_animal, translation_en-de, translation_en-es,
translation_en-fr, word_in_context
```

---

## Prérequis pour Llama-2

Accepter la licence sur https://huggingface.co/meta-llama/Llama-2-13b-chat-hf puis ajouter `HF_TOKEN` dans Kaggle → Add-ons → Secrets.

---

## Citation

```
@misc{wang2024negativeprompt,
      title={NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli},
      author={Xu Wang and Cheng Li and Yi Chang and Jindong Wang and Yuan Wu},
      year={2024},
      eprint={2405.02814},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
