import os 
import sys
import logging
import random
import json

from conllup.conllup import sentenceConllToJson

from utils import PATH_TREEBANKS, PATH_MODELS, PATH_BERTFORDEPREL_VENV, PATH_BERTFORDEPREL_SCRIPT, setup_logging


CONFIG_FILE_PATH = os.path.join(PATH_TREEBANKS, 'treebanks_config.json')


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
    token_count = 0
    selected_sentences = []

    for sentence in all_sentences:

        sentence_json = sentenceConllToJson(sentence)
        num_tokens = len(sentence_json['treeJson']['nodesJson'].values())
        
        if token_count + num_tokens > 5000:
            break
        selected_sentences.append(sentence)
        token_count += num_tokens

    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n\n'.join(selected_sentences) + '\n')



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


def train_all_models(version):

    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as config_file:
        treebanks = json.load(config_file)

    for treebank in treebanks:
        treebank_name = treebank['name']
        total_tokens = treebank['treebank_stats']['total_tokens']
        logging.info(f"Preparing to train model for {treebank_name} with {total_tokens} tokens.")

        if total_tokens < 5000:
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
        print("Usage: start_train.sh <UD_VERSION> [TREEBANK_NAME]")
        sys.exit(1)

    version = sys.argv[1]

    if len(sys.argv) >= 3:
        treebank_name = sys.argv[2]
        try:
            model_folder_path = os.path.join(PATH_MODELS, f"{treebank_name}@{version}")
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
            shuffle_sentences(treebank_name, version)
            train_model(treebank_name, version)
        except Exception as e:
            logging.error(f"Error occurred while training {treebank_name}: {e}")
    else:
        train_all_models(version)


