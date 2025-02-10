import json

import pandas as pd
from transformers import AutoTokenizer

# Load RoBERTa and ChemBERTa tokenizers
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Print vocab sizes
print(f"RoBERTa Vocab Size: {roberta_tokenizer.vocab_size} tokens")
print(f"ChemBERTa Vocab Size: {chemberta_tokenizer.vocab_size} tokens")

# Example text input (RoBERTa optimized for text, ChemBERTa not)
text = "ChemBERTa is based on RoBERTa."

# Tokenize text using both models
roberta_text_tokens = roberta_tokenizer.tokenize(text)
roberta_text_ids = roberta_tokenizer.convert_tokens_to_ids(roberta_text_tokens)

chemberta_text_tokens = chemberta_tokenizer.tokenize(text)
chemberta_text_ids = chemberta_tokenizer.convert_tokens_to_ids(chemberta_text_tokens)

# Example SMILES input (ChemBERTa optimized for this, RoBERTa not)
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

# Tokenize SMILES using both models
roberta_smiles_tokens = roberta_tokenizer.tokenize(smiles)
roberta_smiles_ids = roberta_tokenizer.convert_tokens_to_ids(roberta_smiles_tokens)

chemberta_smiles_tokens = chemberta_tokenizer.tokenize(smiles)
chemberta_smiles_ids = chemberta_tokenizer.convert_tokens_to_ids(chemberta_smiles_tokens)

# Print results
comparison_results = {
    "Text": {
        "Original": text,
        "RoBERTa Tokens": roberta_text_tokens,
        "RoBERTa Token IDs": roberta_text_ids,
        "ChemBERTa Tokens": chemberta_text_tokens,
        "ChemBERTa Token IDs": chemberta_text_ids,
    },
    "SMILES": {
        "Original": smiles,
        "RoBERTa Tokens": roberta_smiles_tokens,
        "RoBERTa Token IDs": roberta_smiles_ids,
        "ChemBERTa Tokens":  chemberta_smiles_tokens,
        "ChemBERTa Token IDs": chemberta_smiles_ids,
    },
}

for (k, r) in comparison_results.items():
    print(k)
    for (i, data) in r.items():
        print(f"{i}: {data}")

