from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Sample text for training
dataset = ["New York", "NewYork", "The quick brown fox", "New York is big"]

# Standard BPE (treats spaces like normal characters)
bpe_tokenizer = Tokenizer(models.BPE())
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Spaces are treated normally
bpe_trainer = trainers.BpeTrainer(vocab_size=50)
bpe_tokenizer.train_from_iterator(dataset, bpe_trainer)

# Byte-Level BPE (BBPE) (preserves spaces explicitly)
bbpe_tokenizer = Tokenizer(models.BPE())
bbpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
bbpe_trainer = trainers.BpeTrainer(vocab_size=50)
bbpe_tokenizer.train_from_iterator(dataset, bbpe_trainer)

# Test tokenization on a sentence
test_sentence = "New York is big"

# Tokenize using Standard BPE
bpe_tokens = bpe_tokenizer.encode(test_sentence).tokens

# Tokenize using Byte-Level BPE (BBPE)
bbpe_tokens = bbpe_tokenizer.encode(test_sentence).tokens

# Display the results
print(f"Test sentence BPE tokens: {bpe_tokens}")
print(f"Test sentence BBPE tokens: {bbpe_tokens}")

print(f"All  BPE tokens:", bpe_tokenizer.get_vocab())
print(f"All BBPE tokens:", bbpe_tokenizer.get_vocab())

