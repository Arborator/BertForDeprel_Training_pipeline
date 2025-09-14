import os

import zipfile
import tarfile
import io
import re
import json 
import sys
import requests
import logging
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

from utils import setup_logging, PATH_TREEBANKS, PATH_MODELS

"""
treebank_utilities.py
This module provides utilities for downloading, extracting, and configuring Universal Dependencies (UD) treebanks and their associated language mappings.
Functions:
    get_treebanks(version, treebanks_download_url):
        Downloads and extracts the UD treebanks archive for a specified version from a given URL.
    extract_ud_languages():
        Scrapes the UD documentation website to extract a mapping of language names to their codes.
    create_treebanks_config_files(version):
        Extracts the treebanks from the downloaded archive, generates configuration information for each treebank, and writes it to a JSON file.
Usage:
    Run this script using update_treebank.sh script providing the UD version and the treebanks download URL as command-line arguments.
    the download URL is https://lindat.mff.cuni.cz/repository/server/api/core/items/55b06337-e49c-4631-9328-b1a38322b1d4/allzip?handleId=11234/1-5901
"""


languages_mapping = {}

def get_treebanks(version, treebanks_download_url):

    response = requests.get(treebanks_download_url)

    if response.status_code == 200:
        treebanks = response.content

        with zipfile.ZipFile(io.BytesIO(treebanks)) as zf:
            treebank_zip_file = f'ud-treebanks-v{version}.tgz'

            if treebank_zip_file in zf.namelist():
                zf.extract(treebank_zip_file, path=PATH_TREEBANKS)
                logging.info(f"Extracted treebank files to {PATH_TREEBANKS}")
            else:
                error_msg = "No treebank files found in the zip archive."
                logging.error(error_msg)
                raise ValueError(error_msg)
    else:
        error_msg = f"Failed to download treebanks. Status code: {response.status_code}"
        logging.error(error_msg)
        raise ConnectionError(error_msg)

def extract_ud_languages():

    ud_docs_url = 'https://quest.ms.mff.cuni.cz/udvalidator/cgi-bin/unidep/langspec/specify_feature.pl'
    response = requests.get(ud_docs_url)

    if response.status_code == 200:
        html_content = response.text
        soup  = BeautifulSoup(html_content, features="lxml")

        language_mapping = {}
        for a in soup.find_all('a'):
            if len(a.text) > 1:
                lang_code = a['href'].split('=')[1]

                if ' ' in a.text:
                    lang_name = f'{a.text.split(" ")[0]}_{a.text.split(" ")[1]}'
                else:
                    lang_name = a.text

                language_mapping[lang_name] = lang_code

        logging.info(f"Extracted {len(language_mapping)} languages from UD documentation.")
        return language_mapping
    else:
        error_msg = f"Failed to fetch UD languages. Status code: {response.status_code}"
        logging.error(error_msg)
        raise ConnectionError(error_msg)


def create_treebanks_config_files(version):
    
    treebanks_archive_path = os.path.join(PATH_TREEBANKS, f'ud-treebanks-v{version}.tgz')
    if not os.path.exists(treebanks_archive_path):
        error_msg = f"Treebank archive {treebanks_archive_path} does not exist."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    languages_mapping = extract_ud_languages()
    treebanks_folder_path = os.path.join(PATH_TREEBANKS, f'ud-treebanks-v{version}')

    with tarfile.open(treebanks_archive_path, 'r:gz') as tar:
        tar.extractall(path=PATH_TREEBANKS)
        
        treebanks_info = []
        for treebank_name in os.listdir(treebanks_folder_path):
            if treebank_name.startswith('UD'):
                print(treebank_name)
                match = re.search(r'_(.*?)-', treebank_name)
                if match:

                    language = match.group(1)
                    language_code = languages_mapping.get(language)
                    model_path = f'ud-models-v{version}/{treebank_name}'

                    treebank_info = {
                        "name": treebank_name,
                        "language": language,
                        "language_code": language_code,
                        "version": version,
                        "treebank_stats": get_treebank_stats(treebank_name),
                        "path_model": os.path.join(PATH_MODELS, model_path)
                    }
                    treebanks_info.append(treebank_info)

        config_file = os.path.join(PATH_TREEBANKS, 'treebanks_config.json')

        with open(config_file, 'w') as f:
            json.dump(treebanks_info, f, indent=4)

        os.remove(treebanks_archive_path)
        
        logging.info(f"Created treebanks configuration file at {config_file}")


def get_treebank_stats(treebank_name):

    treebanks_folder_path = os.path.join(PATH_TREEBANKS, f'ud-treebanks-v{version}')
    treebank_folder_path = os.path.join(treebanks_folder_path, treebank_name)
    stats_path = os.path.join(treebank_folder_path, 'stats.xml')


    tree = ET.parse(stats_path)
    root = tree.getroot()
    size = root.find('size')
    total = size.find('total')

    total_tokens = int(total.find('tokens').text)
    total_sentences = int(total.find('sentences').text)

    treebank_stats = {
        'total_tokens': int(total.find('tokens').text),
        'total_sentences': int(total.find('sentences').text),
        'train_tokens': int(size.find('train/tokens').text),
        'train_sentences': int(size.find('train/sentences').text),
        'dev_tokens': int(size.find('dev/tokens').text),
        'dev_sentences': int(size.find('dev/sentences').text),
        'test_tokens': int(size.find('test/tokens').text),
        'test_sentences': int(size.find('test/sentences').text)
    }
    return treebank_stats
    

if __name__ == "__main__":

    version = sys.argv[1] 
    treebanks_download_url = sys.argv[2] 

    setup_logging()
    logging.info(f"Starting treebank extraction for version {version}")

    get_treebanks(version, treebanks_download_url)
    extract_ud_languages()
    create_treebanks_config_files(version)

    logging.info(f"Treebank extraction and configuration completed for version {version}")




        
    

                


    
    


        
    