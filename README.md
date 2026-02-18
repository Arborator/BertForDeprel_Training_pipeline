# Universal Dependencies (UD) and Surface-Syntactic UD (SUD) Treebank Processing and Parser Training

This repository provides utilities for downloading, extracting, configuring, and training dependency parsers on both Universal Dependencies (UD) and Surface-syntactic Universal Dependencies (SUD) treebanks using the [BertforDeprel architecture](https://github.com/kirianguiller/BertForDeprel).

## Directory Structure

- `treebank_utilities.py` — Download, extract, and configure UD/SUD treebanks.
- `train_utilities.py` — Prepare data and train models for each treebank.
- `statistic_utilities.py` — Collect and summarize model statistics.
- `utils.py` — Shared utilities and environment variable management.
- `start_train.sh` — Shell script to start model training.
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

Run the following command to download and extract UD or SUD treebanks and generate configuration files:

```sh
./update_treebanks.sh <TREEBANK_TYPE> <VERSION> <TREEBANKS_DOWNLOAD_URL>
```

- `<TREEBANK_TYPE>`: Either `UD` or `SUD`
- `<VERSION>`: The treebank version (e.g., 2.17)
- `<TREEBANKS_DOWNLOAD_URL>`: URL to download the treebanks archive or local file path

Examples:
```sh
# Download UD treebanks
./update_treebanks.sh UD 2.17 https://lindat.mff.cuni.cz/repository/server/api/core/items/55b06337-e49c-4631-9328-b1a38322b1d4/allzip?handleId=11234/1-5901

# Use local treebank file for SUD treebanks
./update_treebanks.sh SUD 2.17 /path/to/sud-treebanks-v2.17.tgz
```

### 2. Train Models

After preparing the treebanks, train models using one of the following commands:

```sh
# Train all UD treebanks (default)
./start_train.sh <VERSION>
./start_train.sh <VERSION> UD

# Train all SUD treebanks
./start_train.sh <VERSION> SUD

# Train a specific treebank
./start_train.sh <VERSION> <TREEBANK_NAME>
```

Examples:
```sh
# Train all UD treebanks for version 2.17
./start_train.sh 2.17 UD

# Train all SUD treebanks for version 2.17
./start_train.sh 2.17 SUD

# Train a specific UD treebank
./start_train.sh 2.17 UD_English-Atis

# Train a specific SUD treebank
./start_train.sh 2.17 SUD_French-GSD
```

### 3. Update Model Statistics

After training, update the statistics summary:

```sh
./update_stats.sh
```



