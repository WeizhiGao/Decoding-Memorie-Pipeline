# Anonymous Repo for AAAI 2026 submission: Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection

## Setup
```
cd dmp
conda env create -f environment.yml
conda activate dmp
```

## Model Preparation

You can automatically import models from HuggingFace by uncomment the code in `model/_load_model.py`. If you need offline mode, please first download models with network and replace the path in `model/_load_model.py` with your cache path.

Please use available OpenAI key in `openai_models.py`

## Reproduce Main Results

For LN-Entropy, Lexical Similarity, SelfCheckGPT, and EigenScore, run with the following script
```
cd eigenscore
sh main.sh
```

For Semantic Entropy, run with the following script
```
cd semantic_uncertainty
sh main.sh
```

We appreciate the availability of well-maintained public codebase: [EigenScore](https://github.com/D2I-ai/eigenscore) and [Semantic Entropy](https://github.com/jlko/semantic_uncertainty)
