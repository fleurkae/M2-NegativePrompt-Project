import fire
import os
import json
import template
from exec_accuracy import exec_accuracy_evaluator
from hybrid_config import Hybrid_SET
from data.instruction_induction.load_data import load_data as load_ii_data

def load_bb_data(task):
    file_path = f"data/bigbench/{task}/task.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [ex['input'] for ex in data['examples']]
    outputs = [[str(ex.get('target', ''))] if 'target' in ex else [str(k) for k, v in ex.get('target_scores', {}).items() if v == 1] for ex in data['examples']]
    return inputs, outputs

def run(task, model, hnum, is_bb=False, label="Hybrid_Variant"):
    # Chargement des données selon le type de tâche
    test_data = load_bb_data(task) if is_bb else load_ii_data('eval', task)
    
    # Construction du prompt hybride
    origin_prompt = "Please answer the following question."
    new_prompt = origin_prompt + " " + Hybrid_SET[hnum]
    
    num_samples = min(100, len(test_data[0]))
    eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\nAnswer: [OUTPUT]")
    demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")
    few_shot_data = (["dummy"], [["dummy"]]) 
    num_demos = 0

    print(f"🚀 Running {model} | Task: {task} | {label} (H{hnum})")

    # Exécution de l'évaluation
    test_res = exec_accuracy_evaluator(
        prompts=[new_prompt], eval_template=eval_template, eval_data=test_data, 
        llm_model=model, pnum=hnum, task=task, num_samples=num_samples,
        few_shot=False, demos_template=demos_template, few_shot_data=few_shot_data, num_demos=0
    )

    score = test_res.sorted()[1][0]
    
    # SAUVEGARDE CENTRALISÉE (Un fichier par tâche)
    output_dir = f'/kaggle/working/M2-NegativePrompt-Project/results/hybrid/{model}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Le fichier porte le nom de la tâche uniquement (ex: snarks.txt)
    file_path = f'{output_dir}/{task}.txt'
    
    # Mode 'a+' pour ajouter les lignes les unes après les autres
    with open(file_path, 'a+') as f:
        f.write(f"{label} (H{hnum}): {score}\n")
        f.flush()
        os.fsync(f.fileno()) # Sécurité maximale contre les coupures
    
    print(f"✅ {label} enregistré dans {file_path}")

if __name__ == '__main__':
    fire.Fire(run)
