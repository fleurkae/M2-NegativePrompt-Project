import fire
# On importe le load_data spécifiquement depuis le dossier bigbench
from data.bigbench.load_data import load_data
from exec_accuracy import exec_accuracy_evaluator
from config import Negative_SET
import template
import os
import random

def getPrompt(ori_prompt, num_str):
    new_prompt = ori_prompt
    if num_str > 0:
        # Ajout d'un espace pour que le stimulus ne soit pas collé au prompt
        new_prompt = ori_prompt + " " + Negative_SET[num_str - 1]
    return new_prompt

def run(task, model, pnum, few_shot=False):
    # Chargement des données BIG-Bench
    try:
        # BIG-Bench n'utilise souvent qu'un seul set de données contrairement à Instruction Induction
        test_data = load_data(task)
        induce_data = test_data 
    except TypeError:
        # Au cas où l'auteur aurait structuré le code de la même manière
        test_data = load_data('eval', task)
        induce_data = load_data('induce', task)

    # BIG-Bench n'a pas de prompt dans config.py, on définit un prompt de base universel
    origin_prompt = "Please answer the following question."

    # Reproduit exactement depuis main.py (few-shot setting)
    try:
        few_shot_data = induce_data[0], [random.sample(output, 1)[0] for output in induce_data[1]]
    except Exception:
        few_shot_data = (["dummy"], [["dummy"]])
        
    num_demos = 5
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\n[full_DEMO]\nInput: [INPUT]\nAnswer: [OUTPUT]"
    demos_template = template.DemosTemplate(demos_template)

    # Evaluate on test data
    print('LLM: ', model)
    print('Evaluating on test data...')

    new_prompt = getPrompt(origin_prompt, pnum)
    print('Prompt: ', new_prompt)
    print('Few_shot: ', few_shot)

    # LA FAMEUSE LIMITE DES AUTEURS (Copie exacte de main.py)
    test_num = min(100, len(test_data[0]))

    eval_template = template.EvalTemplate(eval_template)
    
    test_res = exec_accuracy_evaluator(prompts=[new_prompt],
                                    eval_template=eval_template,
                                    eval_data=test_data,
                                    llm_model=model, pnum=pnum,
                                    task=task,
                                    num_samples=test_num,
                                    few_shot=few_shot,
                                    demos_template=demos_template,
                                    few_shot_data=few_shot_data,
                                    num_demos=num_demos)

    test_score = test_res.sorted()[1][0]

    print(f'Test score: {test_score}')

    dir_path = f'results/neg/{model}'
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    with open(f'{dir_path}/{task}.txt', 'a+') as f:
        f.write(f'Test score: {test_score}\n')
        f.write(f'Prompt: {new_prompt}\n')

if __name__ == '__main__':
    fire.Fire(run)
