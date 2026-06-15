import os 
import sys
import logging
import random
import json

from conllup.conllup import sentenceConllToJson

from utils import PATH_TREEBANKS, PATH_MODELS, PATH_BERTFORDEPREL_VENV, PATH_BERTFORDEPREL_SCRIPT, setup_logging


UD_CONFIG_FILE_PATH = os.path.join(PATH_TREEBANKS, 'ud_treebanks_config.json')
SUD_CONFIG_FILE_PATH = os.path.join(PATH_TREEBANKS, 'sud_treebanks_config.json')


def get_config_file_path(treebank_type):
    """Get the config file path based on treebank type (UD or SUD)"""
    if treebank_type == 'SUD':
        return SUD_CONFIG_FILE_PATH
    else:  # Default to UD
        return UD_CONFIG_FILE_PATH


def shuffle_sentences(treebank_name, version):

    ud_folder = os.path.join(PATH_TREEBANKS, f"ud-treebanks-v{version}")
    sud_folder = os.path.join(PATH_TREEBANKS, f"sud-treebanks-v{version}")

    ud_treebank_path = os.path.join(ud_folder, treebank_name)
    sud_treebank_path = os.path.join(sud_folder, treebank_name)

    if os.path.isdir(ud_treebank_path):
        treebank_folder_path = ud_treebank_path
    elif os.path.isdir(sud_treebank_path):
        treebank_folder_path = sud_treebank_path
    else:
        raise FileNotFoundError(
            f"Treebank '{treebank_name}' not found in either {ud_folder} or {sud_folder}")

    all_sentences = []
    for file in os.listdir(treebank_folder_path):

        if file.endswith('.conllu'):
            file_path = os.path.join(treebank_folder_path, file)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                sentences = content.strip().split('\n\n')
                all_sentences.extend(sentences)
    
    random.Random(42).shuffle(all_sentences)

    model_folder_path = os.path.join(PATH_MODELS, f"{treebank_name}@{version}")

    output_file_path = os.path.join( model_folder_path, f"{treebank_name}_train.conllu")
    
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n\n'.join(all_sentences) + '\n')



def train_model(treebank_name, version):
    
    setup_logging()
    logging.info(f"Starting training for treebank: {treebank_name}")

    max_epoch = 64

    model_folder_path = os.path.join(PATH_MODELS, f"{treebank_name}@{version}")
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    

    train_file_path = os.path.join(model_folder_path, f"{treebank_name}_train.conllu")


    TRAINING_CMD = f"{PATH_BERTFORDEPREL_VENV} {PATH_BERTFORDEPREL_SCRIPT} train \
    --new_model_path \"{model_folder_path}\" \
    --ftrain \"{train_file_path}\" \
    --batch_size 16 \
    --gpu_ids 0 \
    --patience 10 \
    --relevant_miscs CpdPos ExtPos \
    --max_epoch {max_epoch}"

    logging.info(f"Executing training command: {TRAINING_CMD}")
    exit_mode = os.system(TRAINING_CMD)
    logging.info(f"Training command exited with code: {exit_mode}")

    path_finished_file = os.path.join(model_folder_path, '.finished')
    if not os.path.exists(path_finished_file):
        error_msg = f"Training did not finish successfully for {treebank_name}. Check the logs."
        logging.error(error_msg)


def train_all_models(version, treebank_type='UD'):
    """Train all models of a specific type (UD or SUD)"""
    setup_logging()
    config_file_path = get_config_file_path(treebank_type)
    
    if not os.path.exists(config_file_path):
        logging.error(f"Config file not found: {config_file_path}")
        return

    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        treebanks = json.load(config_file)

    for treebank in treebanks:
        treebank_name = treebank['name']
        treebank_stats = treebank.get('treebank_stats', {})
        total_tokens = treebank_stats.get('total_tokens', None)
        
        logging.info(f"Preparing to train model for {treebank_name} with {total_tokens if total_tokens else 'unknown'} tokens.")

        if total_tokens is not None and total_tokens < 5000:
            logging.warning(f"Skipping {treebank_name} due to insufficient tokens ({total_tokens}).")
            continue
        else:
            try:
                model_folder_path = os.path.join(PATH_MODELS, f"{treebank_name}@{version}")
                if not os.path.exists(model_folder_path):
                    os.makedirs(model_folder_path)
                    
                shuffle_sentences(treebank_name, version)
                train_model(treebank_name, version)
            except Exception as e:
                logging.error(f"Error occurred while training {treebank_name}: {e}")

    
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: start_train.sh <UD_VERSION> [UD|SUD|TREEBANK_NAME]")
        print("  - ./start_train.sh 2.17           # Train all UD treebanks")
        print("  - ./start_train.sh 2.17 UD        # Train all UD treebanks")
        print("  - ./start_train.sh 2.17 SUD       # Train all SUD treebanks")
        print("  - ./start_train.sh 2.17 TREEBANK  # Train specific treebank")
        sys.exit(1)

    version = sys.argv[1]

    if len(sys.argv) >= 3:
        argument = sys.argv[2]
        
        if argument.upper() == 'UD':
            setup_logging()
            logging.info(f"Training all UD treebanks for version {version}")
            train_all_models(version, 'UD')
        elif argument.upper() == 'SUD':
            setup_logging()
            logging.info(f"Training all SUD treebanks for version {version}")
            train_all_models(version, 'SUD')
        else:
            treebank_name = argument
            try:
                setup_logging()
                logging.info(f"Training specific treebank: {treebank_name}")
                model_folder_path = os.path.join(PATH_MODELS, f"{treebank_name}@{version}")
                if not os.path.exists(model_folder_path):
                    os.makedirs(model_folder_path)
                shuffle_sentences(treebank_name, version)
                train_model(treebank_name, version)
            except Exception as e:
                logging.error(f"Error occurred while training {treebank_name}: {e}")
    else:
        setup_logging()
        logging.info(f"Training all UD treebanks for version {version} (default)")
        train_all_models(version, 'UD')


