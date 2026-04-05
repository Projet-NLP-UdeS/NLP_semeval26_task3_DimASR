# Dimensiomal Aspect-Based Sentiment Analysis
*Refactoring requires the use of `git mv` for continuity*
*Using English for documentation, comments, etc, French for the rest*

#### TODO
- How are hugging face iterations computed? (more control over quick train)
- Save checkpoints during LoRA fine-tuning (2 tasks)
- Save outputs fron Google Collab

## Machine Learning

### Using Google Collab as a Github runtime

1. Code from IDE as usual
2. Push progress to GitHub repo
3. In Collab, open notebook (Ctrl+O) from GitHub :
    - Use `collab_outputs` branch to push results
    - Use https://github.com/Tino-Rg/NLP_semeval26_task3_DimASR/blob/collab_outputs/main.ipynb
4. Save (Ctrl+S) to commit outputs


### Using and maintaining the environment

Using a conda environment enables a more granular control over plateform and package versioning for 
a healthy collaboration workflow. Although resources needed for training might need to be drawn from 
Google Collab, running quick tests in this environment simplifies the workflow and thus accelerates 
development.

- Activating : `conda activate ift714-projet`
- Creation : `conda env create -f environment.yaml`
- Updating : `conda export --from-history --format=environment-yaml > environment.yaml`
(then manually remove `prefix` value and add `pip` specific packages in the required format)

Note : Using Python 3.13 might be [unstable](https://www.python.org/downloads/), switch to 3.12? 
(probably unecessary)

### Running quick tests

I tried to run `bert-base-multilingual-cased` on WSL2 Ubuntu with 8Gb and a Core Ultra 5 125U 
CPU (4.3GHz) using `LR=1e-05, Epochs=1, Batch=32, Dropout=0.1`. Training time estimate was 48s 
per iteration across 88 iterations, for a total 1 hour and 10 minutes (not suitable for quick testing).
WSL also crashed, might need to add swap...

Here are a few useful commands (will detail later) :
```bash
wsl -d Ubuntu -- bash -lc "dmesg | grep -i kill" 
# Check for :
# Killed process XXXX (python)
# Out of memory
rm -rf ~/.vscode-server ~/.vscode-remote
nice -n 10 ionice -c 2 -n 7 python train.py
ulimit -v 6000000
wsl --shutdown
```

Also might try :
```Python
try:
    train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("OOM detected, stopping safely")
```

## Architecture
Based on refactored SemEval-2026 Task 3 [starterkit notebook](https://github.com/DimABSA/DimABSA2026/tree/main/starter_kit/task1).

- `data.py` for data initialization and augmentation
- `train.py` for finetuning and other training operations
- `eval.py` for metrics and benchmark standard
- `models` for models

### Finetuning
https://huggingface.co/docs/transformers/training
