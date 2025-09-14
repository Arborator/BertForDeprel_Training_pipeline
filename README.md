# Universal Dependencies Treebank Processing and Parer Training

This repository provides utilities for downloading, extracting, configuring, training, dependency parsers on Universal Dependencies (UD) treebanks using [BertforDeprel architecture](https://github.com/kirianguiller/BertForDeprel).

## Directory Structure

- `treebank_utilities.py` — Download, extract, and configure UD treebanks.
- `train_utilities.py` — Prepare data and train models for each treebank.
- `statistic_utilities.py` — Collect and summarize model statistics.
- `utils.py` — Shared utilities and environment variable management.
- `start_train.sh` — Shell scriptot  start model training.
- `update_treebanks.sh` — Shell script to update/download treebanks.
- `update_stats.sh` — Shell script to update model statistics.

## Setup

1. **Install dependencies**  
   Make sure you have installed packages in `requirements.txt`

2. **Configure environment**  
   Create a `.env` file with the following variables:
   ```
   PATH_TREEBANKS=/path/to/treebanks
   PATH_MODELS=/path/to/models
   PATH_BERTFORDEPREL_VENV=/path/to/venv/bin/python
   PATH_BERTFORDEPREL_SCRIPT=/path/to/bertfordeprel_script.py
   ```

## Usage

### 1. Download and Prepare Treebanks

Run the following command to download and extract UD treebanks and generate configuration files:

```sh
./update_treebanks.sh <UD_VERSION> <TREEBANKS_DOWNLOAD_URL>
```

- Example:
  ```
  ./update_treebanks.sh 2.16 https://lindat.mff.cuni.cz/repository/server/api/core/items/55b06337-e49c-4631-9328-b1a38322b1d4/allzip?handleId=11234/1-5901
  ```

### 2. Train Models

After preparing the treebanks, train models for all eligible treebanks:

```sh
./start_train.sh <UD_VERSION>
```

### 3. Update Model Statistics

After training, update the statistics summary:

```sh
./update_stats.sh
```



