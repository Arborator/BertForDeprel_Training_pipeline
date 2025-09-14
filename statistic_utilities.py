import json
import os
import logging

from utils import PATH_MODELS, setup_logging


stats_file = os.path.join(PATH_MODELS, 'models_stats.json') 

def get_models_stats():

    models_stats = {}

    for model in os.listdir(PATH_MODELS):
        model_path = os.path.join(PATH_MODELS, model)
        if os.path.isdir(model_path):
            best_scores = os.path.join(model_path, 'scores.best.json')
            print(best_scores)
            if os.path.isfile(best_scores):
                with open(best_scores, 'r', encoding='utf-8') as f:
                    model_stat = json.load(f)
                    models_stats[model] = {
                        'LAS': model_stat['LAS_epoch'],
                        'LAS_chuliu': model_stat['LAS_chuliu_epoch'],
                        'UAS': model_stat['acc_head_epoch'],
                        'train_data': model_stat['training_diagnostics']['data_description']['n_train_sents'],
                        'test_data': model_stat['training_diagnostics']['data_description']['n_test_sents'], 
                    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(models_stats, f, indent=4, ensure_ascii=False)
        logging.info(f"Updated models statistics at {stats_file}")



if __name__ == '__main__':
    setup_logging()
    get_models_stats()

    

            

