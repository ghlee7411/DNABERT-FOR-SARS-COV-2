# 🦠 Learning Representation of DNA sequence of Sars-Cov-2 Virus using DNABERT
- **DNABERT**: https://doi.org/10.1093/bioinformatics/btab083
- **Implementation**: Pytorch, Transformers (Hugging-face), and codes from [DNABERT git-repository](https://github.com/jerryji1993/DNABERT)
- **Toy Dataset**: Reference Sequence of SARS-COV-2 from NCBI Virus (~27kb, ~29000bp)

```python
import logging
logger = logging.getLogger('[SARS-COV-2-DNA-BERT-PRE-TRAIN]')
covid_19_refseq_path = 'input/ncbisarscov2refseq/ncbi_virus_rna_viruses_sars_cov_2_refseq.fasta'
```

# 📝 Rules of DNA Sequence Extraction for Pre-training
- A set of sequences represented as **k-mer tokens** as input (e.g., when k=5, {ATCGGT} -> {ATCGG},{TCGGT})
- The **next sentence prediction (NSP)** of original BERT was **removed** in the DNABERT 
- In the learning of DNABERT, they forced the model to **predict continous *k* tokens** adapting to DNA scenario 
- DNABERT **learns basic syntax and semantics of DNA** via self-supervision, based on **10 to 510 length sequences** extracted from human genome via truncation and sampling (max length of input in DNABERT is 512)
- They randomly masked regions of k *contiguous* tokens that **constitue 15% of the sequence and let DNABERT to predict the masked sequences** based on the remainder, ensuring ample training examples
- Five special tokens (Tokenizer)
    - [CLS]: stands classification token, at the beginning of sequence
    - [PAD]: stands for padding token
    - [UNK]: stands for unknown token
    - [SEP]: stands for separation token, at the end of sequence
    - [MASK]: stands for masked token
- Two approaches for generating data to train: direct non-overlap splitting and random sampling with length of the sequence between 5 to 510
- DNABERT pre-trained 100k steps with masked 15% of k-mers and the last 20k steps they increased the masking rate to 20% (total 120k steps with batch size 2000)
- Model architecture: BERT base, which consists of 12 Transformer layers with 768 hidden units and 12 attention heads in each layer

## 🧫 Process data for pre-training model
```python
from tqdm import tqdm
import os
import random
import numpy as np
```

### Read raw data (DNA sequence of the virus)
```python
def read_refseq_fasta(path):
    seqs = []
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '>': continue
            else:
                seqs.append(line.strip())
    return ''.join(seqs)

refseq = read_refseq_fasta(covid_19_refseq_path)
print('Length of the reference sequence of SARS-COV-2 is {}'.format(len(refseq)))
print(refseq[:2000])
print('...')
print(refseq[-2000:])
```
[Output]
```
Length of the reference sequence of SARS-COV-2 is 29903
ATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCTAAACGAACTTTAAAATCTGTGTGGCTGTCACTCGGCTGCATGCTTAGTGCACTCACGCAGTATAATTAATAACTAATTACTGTCGTTGACAGGACACGAGTAACTCGTCTATCTTCTGCAGGCTGCTTACGGTTTCGTCCGTGTTGCAGCCGATCATCAGCACATCTAGGTTTCGTCCGGGTGTGACCGAAAGGTAAGATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGTGCTCGTACGTGGCTTTGGAGACTCCGTGGAGGAGGTCTTATCAGAGGCACGTCAACATCTTAAAGATGGCACTTGTGGCTTAGTAGAAGTTGAAAAAGGCGTTTTGCCTCAACTTGAACAGCCCTATGTGTTCATCAAACGTTCGGATGCTCGAACTGCACCTCATGGTCATGTTATGGTTGAGCTGGTAGCAGAACTCGAAGGCATTCAGTACGGTCGTAGTGGTGAGACACTTGGTGTCCTTGTCCCTCATGTGGGCGAAATACCAGTGGCTTACCGCAAGGTTCTTCTTCGTAAGAACGGTAATAAAGGAGCTGGTGGCCATAGTTACGGCGCCGATCTAAAGTCATTTGACTTAGGCGACGAGCTTGGCACTGATCCTTATGAAGATTTTCAAGAAAACTGGAACACTAAACATAGCAGTGGTGTTACCCGTGAACTCATGCGTGAGCTTAACGGAGGGGCATACACTCGCTATGTCGATAACAACTTCTGTGGCCCTGATGGCTACCCTCTTGAGTGCATTAAAGACCTTCTAGCACGTGCTGGTAAAGCTTCATGCACTTTGTCCGAACAACTGGACTTTATTGACACTAAGAGGGGTGTATACTGCTGCCGTGAACATGAGCATGAAATTGCTTGGTACACGGAACGTTCTGAAAAGAGCTATGAATTGCAGACACCTTTTGAAATTAAATTGGCAAAGAAATTTGACACCTTCAATGGGGAATGTCCAAATTTTGTATTTCCCTTAAATTCCATAATCAAGACTATTCAACCAAGGGTTGAAAAGAAAAAGCTTGATGGCTTTATGGGTAGAATTCGATCTGTCTATCCAGTTGCGTCACCAAATGAATGCAACCAAATGTGCCTTTCAACTCTCATGAAGTGTGATCATTGTGGTGAAACTTCATGGCAGACGGGCGATTTTGTTAAAGCCACTTGCGAATTTTGTGGCACTGAGAATTTGACTAAAGAAGGTGCCACTACTTGTGGTTACTTACCCCAAAATGCTGTTGTTAAAATTTATTGTCCAGCATGTCACAATTCAGAAGTAGGACCTGAGCATAGTCTTGCCGAATACCATAATGAATCTGGCTTGAAAACCATTCTTCGTAAGGGTGGTCGCACTATTGCCTTTGGAGGCTGTGTGTTCTCTTATGTTGGTTGCCATAACAAGTGTGCCTATTGGGTTCCACGTGCTAGCGCTAACATAGGTTGTAACCATACAGGTGTTGTTGGAGAAGGTTCCGAAGGTCTTAATGACAACCTTCTTGAAATACTCCAAAAAGAGAAAGTCAACATCAATATTGTTGGTGACTTTAAACTTAATGAAGAGATCGCCATTATTTTGGCATCTTTTTCTGCTTCCACAAGTGCTTTTGTGGAAACTGTGAAAGGTTTGGATTATAAAGCATTCAAACAAATTGTTGAATCCTGTGGTAATTTTAAAGTTACAAAAGGAAAAGCTAAAAAAGGTGCCTGGAATATTGGTGAACAGAAATCAATACTGAGTCCTCTTTATGCATTTGCATCAGAGGCTGCTCGTGTTGTACGATCAATTTTCTCCCGCACTCTTGAAACTGCTCAAAATTCTGTGCGTGTTTTACAGAAGGCCGCTATAACAATACTAGATGGAATTTCACAGTATTCACTGA
...
TTGTTTTCTTAGGAATCATCACAACTGTAGCTGCATTTCACCAAGAATGTAGTTTACAGTCATGTACTCAACATCAACCATATGTAGTTGATGACCCGTGTCCTATTCACTTCTATTCTAAATGGTATATTAGAGTAGGAGCTAGAAAATCAGCACCTTTAATTGAATTGTGCGTGGATGAGGCTGGTTCTAAATCACCCATTCAGTACATCGATATCGGTAATTATACAGTTTCCTGTTTACCTTTTACAATTAATTGCCAGGAACCTAAATTGGGTAGTCTTGTAGTGCGTTGTTCGTTCTATGAAGACTTTTTAGAGTATCATGACGTTCGTGTTGTTTTAGATTTCATCTAAACGAACAAACTAAAATGTCTGATAATGGACCCCAAAATCAGCGAAATGCACCCCGCATTACGTTTGGTGGACCCTCAGATTCAACTGGCAGTAACCAGAATGGAGAACGCAGTGGGGCGCGATCAAAACAACGTCGGCCCCAAGGTTTACCCAATAATACTGCGTCTTGGTTCACCGCTCTCACTCAACATGGCAAGGAAGACCTTAAATTCCCTCGAGGACAAGGCGTTCCAATTAACACCAATAGCAGTCCAGATGACCAAATTGGCTACTACCGAAGAGCTACCAGACGAATTCGTGGTGGTGACGGTAAAATGAAAGATCTCAGTCCAAGATGGTATTTCTACTACCTAGGAACTGGGCCAGAAGCTGGACTTCCCTATGGTGCTAACAAAGACGGCATCATATGGGTTGCAACTGAGGGAGCCTTGAATACACCAAAAGATCACATTGGCACCCGCAATCCTGCTAACAATGCTGCAATCGTGCTACAACTTCCTCAAGGAACAACATTGCCAAAAGGCTTCTACGCAGAAGGGAGCAGAGGCGGCAGTCAAGCCTCTTCTCGTTCCTCATCACGTAGTCGCAACAGTTCAAGAAATTCAACTCCAGGCAGCAGTAGGGGAACTTCTCCTGCTAGAATGGCTGGCAATGGCGGTGATGCTGCTCTTGCTTTGCTGCTGCTTGACAGATTGAACCAGCTTGAGAGCAAAATGTCTGGTAAAGGCCAACAACAACAAGGCCAAACTGTCACTAAGAAATCTGCTGCTGAGGCTTCTAAGAAGCCTCGGCAAAAACGTACTGCCACTAAAGCATACAATGTAACACAAGCTTTCGGCAGACGTGGTCCAGAACAAACCCAAGGAAATTTTGGGGACCAGGAACTAATCAGACAAGGAACTGATTACAAACATTGGCCGCAAATTGCACAATTTGCCCCCAGCGCTTCAGCGTTCTTCGGAATGTCGCGCATTGGCATGGAAGTCACACCTTCGGGAACGTGGTTGACCTACACAGGTGCCATCAAATTGGATGACAAAGATCCAAATTTCAAAGATCAAGTCATTTTGCTGAATAAGCATATTGACGCATACAAAACATTCCCACCAACAGAGCCTAAAAAGGACAAAAAGAAGAAGGCTGATGAAACTCAAGCCTTACCGCAGAGACAGAAGAAACAGCAAACTGTGACTCTTCTTCCTGCTGCAGATTTGGATGATTTCTCCAAACAATTGCAACAATCCATGAGCAGTGCTGACTCAACTCAGGCCTAAACTCATGCAGACCACACAAGGCAGATGGGCTATATAAACGTTTTCGCTTTTCCGTTTACGATATATAGTCTACTCTTGTGCAGAATGAATTCTCGTAACTACATAGCACAAGTAGATGTAGTTAACTTTAATCTCACATAGCAATCTTTAATCAGTGTGTAACATTAGGGAGGACTTGAAAGAGCCACCACATTTTCACCGAGGCCACGCGGAGTACGATCGAGTGTACAGTGAACAATGCTAGGGAGAGCTGCCTATATGGAAGAGCCCTAATGTGTAAAATTAATTTTAGTAGTGCTATCCCCATGTGATTTTAATAGCTTCTTAGGAGAATGACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
```

### Sequence Sampling
```python
def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string)-kmer):
        sequence.append(original_string[i:i+kmer])
    
    sequence.append(original_string[-kmer:])
    return sequence


def get_sequence_sampling_positions(sequence_length, kmer, sampling_rate, sample_length):
    num_samples = int(sequence_length * sampling_rate / sample_length)
    starts, ends = [], []
    for _ in range(num_samples):
        start = np.random.randint(sequence_length - sample_length - kmer)
        end   = start + sample_length
        starts.append(start)
        ends.append(end)
    return starts, ends


def sampling(refseq, kmer, num_samples, sampling_rate, min_sample_length, max_sample_length):
    refseq_len = len(refseq)
    pretrain_sequences = []
    per = 0
    while len(pretrain_sequences) < num_samples:
        sample_length = np.random.randint(min_sample_length, max_sample_length)
        starts, ends = get_sequence_sampling_positions(refseq_len, kmer, sampling_rate, sample_length)
        seqs = [refseq[start:end+1] for start, end in zip(starts, ends)]
        seqs = [get_kmer_sequence(s, kmer) for s in seqs]
        pretrain_sequences += seqs
        
        if len(pretrain_sequences) > (num_samples * per / 100):
            print('Sampling ... {:>8} / {:>8} ({}%)'.format(len(pretrain_sequences), num_samples, per))
            per += 5
    
    random.shuffle(pretrain_sequences)
    return pretrain_sequences[:num_samples]

raw_sequence_sampling = False

kmer = 3
sampling_rate = 1.0
min_sample_length = 10
max_sample_length = 510
num_samples = 100000
covid_19_refseq_train_seqs = 'ncbi_virus_rna_viruses_sars_cov_2_refseq_seq.train.txt'
covid_19_refseq_test_seqs = 'ncbi_virus_rna_viruses_sars_cov_2_refseq_seq.test.txt'

if raw_sequence_sampling or not os.path.exists(covid_19_refseq_train_seqs):
    # train
    pretrain_seqs = sampling(refseq, kmer, num_samples, sampling_rate, min_sample_length, max_sample_length)
    with open(covid_19_refseq_train_seqs, 'w') as f:
        for pretrain_seq in pretrain_seqs:
            seq = ' '.join(pretrain_seq)
            f.write('{}\n'.format(seq))
    
    # test
    pretrain_seqs = sampling(refseq, kmer, int(num_samples * 0.1), sampling_rate, min_sample_length, max_sample_length)
    with open(covid_19_refseq_test_seqs, 'w') as f:
        for pretrain_seq in pretrain_seqs:
            seq = ' '.join(pretrain_seq)
            f.write('{}\n'.format(seq))
```
[Output]
```
Sampling ...      137 /   100000 (0%)
Sampling ...     5045 /   100000 (5%)
Sampling ...    10052 /   100000 (10%)
Sampling ...    15005 /   100000 (15%)
Sampling ...    20041 /   100000 (20%)
Sampling ...    25232 /   100000 (25%)
Sampling ...    30113 /   100000 (30%)
Sampling ...    35053 /   100000 (35%)
Sampling ...    40002 /   100000 (40%)
Sampling ...    45254 /   100000 (45%)
Sampling ...    50001 /   100000 (50%)
Sampling ...    55177 /   100000 (55%)
Sampling ...    60315 /   100000 (60%)
Sampling ...    65393 /   100000 (65%)
Sampling ...    70074 /   100000 (70%)
Sampling ...    75046 /   100000 (75%)
Sampling ...    81243 /   100000 (80%)
Sampling ...    85024 /   100000 (85%)
Sampling ...    92833 /   100000 (90%)
Sampling ...    95700 /   100000 (95%)
Sampling ...   101569 /   100000 (100%)
Sampling ...      115 /    10000 (0%)
Sampling ...      668 /    10000 (5%)
Sampling ...     1114 /    10000 (10%)
Sampling ...     1592 /    10000 (15%)
Sampling ...     2032 /    10000 (20%)
Sampling ...     2512 /    10000 (25%)
Sampling ...     3512 /    10000 (30%)
Sampling ...     3659 /    10000 (35%)
Sampling ...     4086 /    10000 (40%)
Sampling ...     4506 /    10000 (45%)
Sampling ...     5013 /    10000 (50%)
Sampling ...     6294 /    10000 (55%)
Sampling ...     6376 /    10000 (60%)
Sampling ...     6508 /    10000 (65%)
Sampling ...     8714 /    10000 (70%)
Sampling ...     9864 /    10000 (75%)
Sampling ...    10119 /    10000 (80%)
```

### Tokenizing - k-mer DNATokenizer
- In the DNABERT, they prepared vocab files for DNA-Tokenizer, which is called *PreTrainedTokenizer*
- Basically, vocab file format is same with BERT base Tokenizer (see [k=3 vocabs of DNABERT](https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-3/vocab.txt))

```python
import itertools
vocabs = []
special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
DNA = ['A', 'T', 'C', 'G']
k_mer_tokens = sorted([''.join(bases) for bases in itertools.product(DNA, repeat=kmer)])
vocabs = special_tokens + k_mer_tokens
vocab_file_name = 'dna-{}-mer-vocab.txt'.format(kmer)
with open(vocab_file_name, 'w') as f:
    for vocab in vocabs:
        f.write('{}\n'.format(vocab))
        
print(vocabs, len(vocabs))
```
[Output]
```
['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT'] 69
```

```python
from tokenization_dna import DNATokenizer

dna_tokenizer = DNATokenizer(vocab_file=vocab_file_name)
dna_tokenizer.save_pretrained('dnabert-base-tokenizer') # not working? why?
dna_tokenizer
```
[Output]
```
<tokenization_dna.DNATokenizer at 0x7f55bea30a50>
```

### Prepare Dataset
```python
import pickle
import torch

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from multiprocessing import Pool

def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=True):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, max_length=max_length)["input_ids"]
    return examples

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, model_type, n_process, overwrite_cache, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            
            if n_process == 1:
                self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
            else:
                n_proc = n_process
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(lines)/n_proc)
                for i in range(1, n_proc+1):
                    if i != n_proc:
                        indexes.append(len_slice*(i))
                    else:
                        indexes.append(len(lines))
                results = []
                for i in range(n_proc):
                    results.append(p.apply_async(convert_line_to_example,[tokenizer, lines[indexes[i]:indexes[i+1]], block_size,]))
                    print(str(i) + " start")
                p.close() 
                p.join()

                self.examples = []
                for result in results:
                    ids = result.get()
                    self.examples.extend(ids)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
    
    
model_type = 'dnabert-pretrain'
n_process = 4
block_size = 512
overwrite_cache = False
train_dataset = LineByLineTextDataset(dna_tokenizer, model_type, n_process, overwrite_cache, covid_19_refseq_train_seqs, block_size)
test_dataset  = LineByLineTextDataset(dna_tokenizer, model_type, n_process, overwrite_cache, covid_19_refseq_test_seqs, block_size)
```
[Output]
```
0 start
1 start
2 start
3 start
0 start
1 start
2 start
3 start
```

```python
from torch.nn.utils.rnn import pad_sequence

batch_size = 16
local_rank = -1

def collate(examples):
    tokenizer = dna_tokenizer
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              collate_fn=collate)

test_sampler = RandomSampler(test_dataset) if local_rank == -1 else DistributedSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, 
                             sampler=test_sampler,
                             batch_size=batch_size, 
                             collate_fn=collate)
```

## 🔬 BERT Modeling
### Load BERT base model architecture
```python
from transformers import WEIGHTS_NAME, AdamW, BertConfig, BertForMaskedLM

config = BertConfig()
config.num_hidden_layers = 3
config.vocab_size = dna_tokenizer.vocab_size
model = BertForMaskedLM(config=config)
```
``` python
# summary
print('='*100)
print('')
print(config)
print('-'*100)
for name, tensor in model.named_parameters():
    print(name, tensor.shape)
    print('-'*100)
print('')
print('Total number of parameters: {}'.format(model.num_parameters()))
print('')
print('='*100)
```
[Output]
```
====================================================================================================

BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 3,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.6.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 69
}

----------------------------------------------------------------------------------------------------
bert.embeddings.word_embeddings.weight torch.Size([69, 768])
----------------------------------------------------------------------------------------------------
bert.embeddings.position_embeddings.weight torch.Size([512, 768])
----------------------------------------------------------------------------------------------------
bert.embeddings.token_type_embeddings.weight torch.Size([2, 768])
----------------------------------------------------------------------------------------------------
bert.embeddings.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.embeddings.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.self.query.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.self.key.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.self.value.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.output.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.intermediate.dense.bias torch.Size([3072])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.output.dense.weight torch.Size([768, 3072])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.output.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.output.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.0.output.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.self.query.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.self.query.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.self.key.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.self.key.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.self.value.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.self.value.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.output.dense.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.output.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.output.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.attention.output.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.intermediate.dense.weight torch.Size([3072, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.intermediate.dense.bias torch.Size([3072])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.output.dense.weight torch.Size([768, 3072])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.output.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.output.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.1.output.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.self.query.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.self.query.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.self.key.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.self.key.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.self.value.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.self.value.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.output.dense.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.output.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.output.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.attention.output.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.intermediate.dense.weight torch.Size([3072, 768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.intermediate.dense.bias torch.Size([3072])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.output.dense.weight torch.Size([768, 3072])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.output.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.output.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
bert.encoder.layer.2.output.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
cls.predictions.bias torch.Size([69])
----------------------------------------------------------------------------------------------------
cls.predictions.transform.dense.weight torch.Size([768, 768])
----------------------------------------------------------------------------------------------------
cls.predictions.transform.dense.bias torch.Size([768])
----------------------------------------------------------------------------------------------------
cls.predictions.transform.LayerNorm.weight torch.Size([768])
----------------------------------------------------------------------------------------------------
cls.predictions.transform.LayerNorm.bias torch.Size([768])
----------------------------------------------------------------------------------------------------

Total number of parameters: 22305093

====================================================================================================
```

### Init optimizer and scheduler
```python
# Prepare optimizer and schedule (linear warmup and decay)
from transformers import get_linear_schedule_with_warmup

num_train_epochs = 10
gradient_accumulation_steps = 1
t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
weight_decay = 0.0
learning_rate = 5e-5
adam_epsilon = 1e-8
beta1 = 0.9
beta2 = 0.99
warmup_steps = 10

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, betas=(beta1,beta2))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
```

```python
# Check if saved optimizer or scheduler states exist
model_name_or_path = 'dnabert_pretrain'

if (model_name_or_path
    and os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt"))
    and os.path.isfile(os.path.join(model_name_or_path, "scheduler.pt"))):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))
```

## 🛠 Pre-training
```python
from copy import deepcopy

# Setup CUDA, GPU & distributed training
if local_rank == -1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl")
    n_gpu = 1
device = device
print(device)
```
[Output]
```
cuda:0
```

```python
MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    
    mask_list = MASK_LIST[tokenizer.kmer]

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_index==1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True
    

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
```

```python
model.to(device)
```
[Output]
```
BertForMaskedLM(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(69, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (2): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (cls): BertOnlyMLMHead(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
      (decoder): Linear(in_features=768, out_features=69, bias=True)
    )
  )
)
```

## 🔥 Let's burn 🔥
```python
from tqdm import tqdm, trange


mlm = True
mlm_probability = 0.15
output_dir = 'dna-bert-sars-cov-2-chkpt'
os.makedirs(output_dir, exist_ok=True)
logging_step = 25
best_test_loss = 1e+10

model.zero_grad()
for e in range(num_train_epochs):
    
    train_global_step = 0
    test_global_step = 0
    train_loss = 0
    test_loss = 0
    
    # Train
    for step, batch in enumerate(train_dataloader):
        
        # Inference
        inputs, labels = mask_tokens(batch, dna_tokenizer, mlm_probability) if mlm else (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.train()
        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        
        # Optimize
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        train_global_step += 1
        
        if train_global_step % logging_step == 0:
            print('epoch {:>5} \t train loss: {:>.6f} \t step: {:>6} / {:>6}'.format(e, 
                                                                                     train_loss/train_global_step, 
                                                                                     train_global_step, 
                                                                                     len(train_dataloader)))
        
        
        
    # Test
    for step, batch in enumerate(test_dataloader):
        
        # Inference
        inputs, labels = mask_tokens(batch, dna_tokenizer, mlm_probability) if mlm else (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        outputs = model(inputs, labels=labels)
        loss = outputs[0]
        test_loss += loss.item()
        test_global_step += 1
        
        if test_global_step % logging_step == 0:
            print('epoch {:>5} \t test loss: {:>.6f} \t step: {:>6} / {:>6}'.format(e, 
                                                                                    test_loss/test_global_step, 
                                                                                    test_global_step, 
                                                                                    len(test_dataloader)))
        
        
        
    
    mean_test_loss = test_loss / test_global_step     
    if mean_test_loss < best_test_loss:
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        dna_tokenizer.save_pretrained(output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        print('>> model updated (best loss: {:>.7f} to {:>.7f})'.format(best_test_loss, mean_test_loss))
        best_test_loss = mean_test_loss
        
```




