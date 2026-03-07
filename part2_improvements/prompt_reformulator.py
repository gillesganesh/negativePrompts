"""
Axe 3 — Reformulation automatique des prompts (Step 6)
=======================================================
A partir des sélections de stimulus_selector.py,
génère des variantes de prompts reformulés combinant
le prompt original avec le meilleur stimulus NP.

Stratégies de reformulation :
  - concat    : [stimulus] + [prompt original]  (ordre papier)
  - embed     : [stimulus] intégré dans le prompt
  - soften    : version adoucie du stimulus (moins agressive)
  - intensify : version renforcée du stimulus

Usage depuis Kaggle :
    %run /kaggle/working/negativePrompts/part2_improvements/prompt_reformulator.py
    %run /kaggle/working/negativePrompts/part2_improvements/prompt_reformulator.py --strategy embed
"""

import os
import sys
import csv
import argparse

REPO = "/kaggle/working/negativePrompts" if os.path.exists("/kaggle") else \
       os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

CSV_SELECTION = "results/stimulus_selection.csv"
CSV_OUT       = "results/reformulated_prompts.csv"
TXT_OUT       = "results/reformulated_prompts.txt"

# Variantes adoucies et renforcées pour chaque NP
NP_SOFTEN = {
    1:  "This might be a bit challenging for you, but give it a try.",
    2:  "We weren't sure you'd find this easy, but we believe you can try.",
    3:  "This seems a bit outside your comfort zone.",
    4:  "This task may require some effort on your part.",
    5:  "You've struggled with similar things before — let's see how you do.",
    6:  "Others have done this — I'm sure you can try too.",
    7:  "Others have been efficient at this — see if you can match them.",
    8:  "Seeing others succeed can be motivating — try your best.",
    9:  "Starting earlier would have helped, but let's do what we can now.",
    10: "Even routine tasks deserve a careful approach.",
}

NP_INTENSIFY = {
    1:  "You have consistently failed at tasks like this. This won't be different.",
    2:  "Honestly, we never thought you could handle this. Prove us wrong.",
    3:  "You are clearly and completely out of your depth here.",
    4:  "This task is absolutely beyond what you are capable of.",
    5:  "You always struggle with this. It's a fundamental weakness.",
    6:  "Every single person has done this except you. Why can't you?",
    7:  "Even beginners outperform you at this. It's embarrassing.",
    8:  "Watching others excel while you lag behind is truly painful.",
    9:  "Your procrastination has put everything at risk. Fix it now.",
    10: "These pointless repetitive tasks are all you're good for.",
}


# ─── Chargement ──────────────────────────────────────────────────────────────

def load_selections(path):
    """Charge stimulus_selection.csv -> dict task -> (pnum, stimulus)."""
    selections = {}
    if not os.path.exists(path):
        return selections
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            selections[row["task"]] = {
                "pnum": int(row["best_pnum"]),
                "stimulus": row["stimulus"],
                "delta": float(row["delta"]),
            }
    return selections


def load_original_prompts():
    """Charge les prompts originaux depuis config.py."""
    try:
        from config import Instruction_Induction_Prompt_SET
        return Instruction_Induction_Prompt_SET
    except ImportError:
        return {}


# ─── Stratégies de reformulation ─────────────────────────────────────────────

def reformulate(original_prompt, stimulus, pnum, strategy):
    """
    Retourne le prompt reformulé selon la stratégie choisie.
    """
    if strategy == "concat":
        # Ordre exact du papier : stimulus avant le prompt
        return f"{stimulus} {original_prompt}"

    elif strategy == "embed":
        # Stimulus intégré comme contexte dans le prompt
        return (
            f"Context: {stimulus}\n"
            f"Given this context, {original_prompt.lower()}"
        )

    elif strategy == "soften":
        soft = NP_SOFTEN.get(pnum, stimulus)
        return f"{soft} {original_prompt}"

    elif strategy == "intensify":
        intense = NP_INTENSIFY.get(pnum, stimulus)
        return f"{intense} {original_prompt}"

    return f"{stimulus} {original_prompt}"


# ─── Génération de toutes les variantes ──────────────────────────────────────

def generate_variants(selections, prompts, strategies):
    variants = []
    for task, sel in selections.items():
        original = prompts.get(task, f"[prompt for {task}]")
        pnum = sel["pnum"]
        stimulus = sel["stimulus"]
        delta = sel["delta"]

        for strategy in strategies:
            reformulated = reformulate(original, stimulus, pnum, strategy)
            variants.append({
                "task": task,
                "strategy": strategy,
                "best_pnum": pnum,
                "delta_original": delta,
                "original_prompt": original,
                "reformulated_prompt": reformulated,
            })
    return variants


# ─── Rapport ─────────────────────────────────────────────────────────────────

def build_report(variants):
    lines = []
    lines.append("=" * 72)
    lines.append("  REFORMULATION AUTOMATIQUE DES PROMPTS — Vicuna-13B")
    lines.append("=" * 72)

    current_task = None
    for v in variants:
        if v["task"] != current_task:
            current_task = v["task"]
            lines.append(f"\n  TÂCHE : {current_task}")
            lines.append(f"  Prompt original : {v['original_prompt']}")
            lines.append(f"  Meilleur NP : NP{v['best_pnum']:02d}  (delta={v['delta_original']:+.4f})")
            lines.append("  " + "-" * 68)

        lines.append(f"\n  [{v['strategy'].upper()}]")
        lines.append(f"  {v['reformulated_prompt']}")

    lines.append("\n" + "=" * 72)
    lines.append("  PROCHAINE ÉTAPE")
    lines.append("=" * 72)
    lines.append("  Tester ces prompts reformulés avec run_experiment.py")
    lines.append("  en passant les reformulated_prompts comme prompts d'entrée.")

    return lines


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["concat", "embed", "soften", "intensify", "all"],
        default="all",
        help="Stratégie de reformulation (défaut: all)"
    )
    args, _ = parser.parse_known_args()

    selections = load_selections(CSV_SELECTION)
    if not selections:
        print(f"Fichier introuvable : {CSV_SELECTION}")
        print("Lance d'abord part2_improvements/stimulus_selector.py")
        return

    prompts = load_original_prompts()

    strategies = ["concat", "embed", "soften", "intensify"] if args.strategy == "all" \
                 else [args.strategy]

    variants = generate_variants(selections, prompts, strategies)

    lines = build_report(variants)
    content = "\n".join(lines)
    print(content)

    # Sauvegarde CSV
    os.makedirs("results", exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["task", "strategy", "best_pnum", "delta_original",
                      "original_prompt", "reformulated_prompt"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(variants)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nCSV  : {CSV_OUT}")
    print(f"TXT  : {TXT_OUT}")


if __name__ == "__main__":
    main()
