# RoBERTa-tiny-cased

A small case-preserving RoBERTa model pre-trained for production use. Feel free to download our model from [Baidu Disk](https://pan.baidu.com/s/1gIawiCKqUz-QzdHas0swVA) (Extraction Code: yhmq) or [Google Drive](https://drive.google.com/file/d/1oHH4nmXVGQRdyryqqcLdyoYvZR-sGRWp/view?usp=sharing) or from [HuggingFace model hub](https://huggingface.co/haisongzhang/roberta-tiny-cased).

### Model Parameters

|              | Layers | Hidden Size | #Heads | #Parameters |
| ------------ | ------ | ----------- | ------ | ----------- |
| RoBERTa-tiny | 4      | 512         | 8      | 28M         |

### Pre-training Data

We used a 43G corpus consists of [Wikipedia](https://dumps.wikimedia.org/enwiki/latest/), [BookCorpus](https://yknzhu.wixsite.com/mbweb) and [UMBC WebBase Corpus](https://ebiquity.umbc.edu/resource/html/id/351). Except BookCorpus, all pre-training data is **case-preserving**.

| Corpus Name         | Corpus Size | Domain               | #Sentences | #Words |
| ------------------- | ----------- | -------------------- | ---------- | ------ |
| Wikipedia           | 21G         | wiki                 | ~212M      | ~4B    |
| BookCorpus          | 4.4G        | fiction, story, etc. | ~74M       | ~1B    |
| UMBC WebBase Corpus | 18G         | web pages            | ~180M      | ~3B    |

### Pre-training Procedure

We used code from [Transformers](https://github.com/huggingface/transformers) to pre-train RoBERTa-tiny. [Datasets](https://github.com/huggingface/datasets) library was used to provide fast and efficient access to disk data. During pre-training, we followed the setting from [RoBERTa](https://arxiv.org/abs/1907.11692) and only used MLM loss as pre-training objective. However, we used Wordpiece as tokenizer, while BPE was used in original RoBERTa. Input data was organized in FULL-SENTENCES format. The whole pre-training took about 5 days on 8 V100 GPUs for 20 epochs.

Here we list some important hyperparameters:

| Initial Learning Rate | Epochs/Steps | Batch Size | Maximum Length |
| --------------------- | ------------ | ---------- | -------------- |
| 1e-4                  | 20/~1.8M     | 512        | 256            |

### Results

We fine-tuned our RoBERTa-tiny (cased) model on all tasks from [GLUE](https://gluebenchmark.com/) (Task descriptions are listed below), and compared the test set results with BERT-small, an **uncased BERT model** **with same structure** released by [Google](https://github.com/google-research/bert). 

|                         | CoLA                                       | SST-2                      | MRPC                                  | STS-B                                | QQP                                   | MNLI                 | QNLI                 | RTE                  | WNLI                 |
| ----------------------- | ------------------------------------------ | -------------------------- | ------------------------------------- | ------------------------------------ | ------------------------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| Task Description        | Classification (grammatical acceptability) | Classification (sentiment) | Classification (semantic equivalence) | Classification (semantic similarity) | Classification (semantic consistency) | Classification (NLI) | Classification (NLI) | Classification (NLI) | Classification (NLI) |
| #Sentences for Training | 8551                                       | 67349                      | 3668                                  | 5749                                 | 363870                                | 392702               | 104743               | 2490                 | 635                  |
| Average Input Length    | 11.5                                       | 14.0                       | 54.7                                  | 29.1                                 | 31.4                                  | 40.8                 | 50.9                 | 68.0                 | 37.5                 |

| Model                          | Overall  | CoLA     | SST-2    | MRPC          | STS-B         | QQP           | MNLI-m   | MNLI-mm  | QNLI     | RTE      | WNLI     |
| ------------------------------ | -------- | -------- | -------- | ------------- | ------------- | ------------- | -------- | -------- | -------- | -------- | -------- |
| BERT-small (**uncased**)       | 71.2     | 27.8     | 89.7     | 83.4/76.2     | 78.8/77.0     | 68.1/87.0     | 77.6     | 77.0     | **86.4** | 61.8     | 62.3     |
| RoBERTa-tiny (**cased**, ours) | **74.0** | **35.9** | **89.8** | **86.2/81.8** | **83.8/82.7** | **68.9/88.2** | **77.7** | **77.2** | 85.9     | **66.5** | **65.1** |

For RTE, STS, MRPC and QNLI, we found it helpful to finetune starting from the MNLI single-task model, rather than the baseline pretrained RoBERTa. For each task, we selected the best fine-tuning hyperparameters from the lists below, and trained for 4 epochs:

- batch sizes: 8, 16, 32, 64, 128
- learning rates: 3e-4, 1e-4, 5e-5, 3e-5

### Use Our Model

Our pre-trained model is specially suitable for low latency applications. Combined with knowledge distillation and task-specific fine-tuning, our model can achieve high inference speed while keeping similar performance with larger models. HuggingFace Transformers is recommended when loading our model for further fine-tuning:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("haisongzhang/roberta-tiny-cased")

model = AutoModelWithLMHead.from_pretrained("haisongzhang/roberta-tiny-cased")
```

**Note**: When loading the tokenizer in transformers, use BertTokenizer instead of RobertaTokenizer since Wordpiece was used in this model. 

### Acknowledgement

This work was done by my intern [@raleighhan](https://github.com/RaleighHan)(撖朝润) during his internship at NLP Group of Tencent AI Lab.

### References

HuggingFace Transformers: https://github.com/huggingface/transformers

BERT: Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

RoBERTa: Liu Y, Ott M, Goyal N, et al. Roberta: A robustly optimized bert pretraining approach[J]. arXiv preprint arXiv:1907.11692, 2019.

BERT-small: Turc I, Chang M W, Lee K, et al. Well-read students learn better: On the importance of pre-training compact models[J]. arXiv preprint arXiv:1908.08962, 2019.

