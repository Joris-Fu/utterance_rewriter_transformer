# Utterance rewriter with transformer

## Requirements
* python==3.x (Let's move on to python 3 if you still use python 2)
* tensorflow==1.12.0
* tqdm>=4.28.1
* jieba>=0.3x
* sumeval>=0.2.0

## Model Structure
### Based
Model is based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
We implemented the transformer version of [Improving Multi-turn Dialogue Modelling with Utterance ReWriter](https://arxiv.org/abs/1906.07004). The author also provide LSTM version at https://github.com/chin-gyou/dialogue-utterance-rewriter, recommand to check it out!

# PS
The project is under construction. We will update the evaluation result when finished!

| name | type | detail |
|--------------------|------|-------------|
vocab_size | int | vocab size
train | str | train dataset dir
eval | str| eval dataset dir
test | str| data for calculate rouge score
vocab | str| vocabulary file path
batch_size | int| train batch size
eval_batch_size | int| eval batch size
lr | float| learning rate
warmup_steps | int| warmup steps by learing rate
logdir | str| log directory
num_epochs | int| the number of train epoch
evaldir | str| evaluation dir
d_model | int| hidden dimension of encoder/decoder
d_ff | int| hidden dimension of feedforward layer
num_blocks | int| number of encoder/decoder blocks
num_heads | int| number of attention heads
maxlen1 | int| maximum length of a source sequence
maxlen2 | int| maximum length of a target sequence
dropout_rate | float| dropout rate
beam_size | int| beam size for decode
gpu_nums | int| gpu amount, which can allow how many gpu to train this model， default 1
