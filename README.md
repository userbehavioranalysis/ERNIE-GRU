# Experiment Code Documentation 

## Environment Configuration

### 1. LLM-specific Environment (Env_For_LLM)
```yaml
Config file: Env_For_LLM(anaconda).yaml
Features:
- Supports LLM inference
- Integrates langchain library
```

### 2. ERNIE/BERT Environment (Env_For_ErnieBert)
```yaml
Config file: Env_For_ErnieBert(anaconda).yaml
Features:
- Supports ERNIE/BERT model training
- Integrates PaddlePaddle framework
```

## Code Structure
```
code-submit/
├── data_and_checkpoints/        # Experimental data & model checkpoints
│   ├── test.pkl                 # Preprocessed test dataset
│   └── test.txt                 # Raw test data
├── prompt_teng.py               # Prompt engineering template generator
├── Ernie.ipynb                  # ERNIE model experiments
├── Bert.ipynb                   # BERT comparative experiments
├── LLM.ipynb                    # LLM fine-tuning experiments
├── label_dealing_for_LLM.ipynb  # Label processing pipeline for LLM
└── model_experiment_results.csv # Model experiment results
```

## Workflow
1. LLM experiments:
```bash
conda activate Env_For_LLM
label_dealing_for_LLM.ipynb
LLM.ipynb
```

2. ERNIE/BERT experiments:
```bash
# ERNIE experiments
conda activate Env_For_ErnieBert
Ernie.ipynb

# BERT experiments
conda activate Env_For_ErnieBert 
Bert.ipynb
```

## Important Notes
1. Maintain environment isolation to avoid dependency conflicts
2. Execute data preprocessing first
3. Experiment parameters are configured in notebook header cells
