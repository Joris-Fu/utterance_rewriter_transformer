# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Last modified by fu shiding
fsd.joris@hotmail.com
'''
import tensorflow as tf
from utils import calc_num_batches

def _load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = []
    with open(vocab_fpath, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.replace('\n', ''))
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}

    return token2idx, idx2token

def load_stop(vocab_path):
    """
    load stop word
    :param vocab_path: stop word path
    :return: stop word list
    """
    stop_words = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.replace('\n', ''))

    return sorted(stop_words, key=lambda i: len(i), reverse=True)

def _load_data(fpaths, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    with open(fpaths, 'r', encoding='utf-8') as f:
        for line in f:
            splits = line.split('\t\t')
            if len(splits) != 2: continue
            sen1 = splits[0].replace('\n', '').strip()
            sen2 = splits[1].replace('\n', '').strip()
            if len(list(sen1)) + 1 > maxlen1-8: continue
            if len(list(sen2)) + 1 > maxlen2-4: continue

            sents1.append(sen1.encode('utf-8'))
            sents2.append(sen2.encode('utf-8'))

    return sents1[:400000], sents2[:400000]

def _encode(inp, token2idx, maxlen, type, maxlen_utterance=30):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp = inp.decode('utf-8')
    if type == 'x':
        ## 处理一下输入，每轮对话用<del>,原句子使用|分割,历史和当前句用<sep>分割,原本使用tab分割
        new_inp = []
        for ch in inp.split("\t")[0]:
            if ch=="|":
                new_inp.append("<sep>")
            else:
                new_inp.append(ch)

        tokens = ['<s>'] + new_inp + ['</s>']
        while len(tokens) < maxlen:
            tokens.append('<pad>')
        tokens += ['sep']+['<s>'] + list(inp.split("\t")[1]) + ['</s>']
        while len(tokens) < maxlen+maxlen_utterance:
            tokens.append('<pad>')
        return [token2idx.get(token, token2idx['<unk>']) for token in tokens]

    else:
        inputs = ['<s>'] + list(inp)
        target = list(inp) + ['</s>']
        while len(target) < maxlen:
            inputs.append('<pad>')
            target.append('<pad>')
        return [token2idx.get(token, token2idx['<unk>']) for token in inputs], [token2idx.get(token, token2idx['<unk>']) for token in target]

def _generator_fn(sents1, sents2, vocab_fpath, maxlen1, maxlen2):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = _load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = _encode(sent1, token2idx, maxlen1, "x", maxlen2)
        # 加入turn ids
        turn_id = 1
        turn_ids = []
        for id in x:
            if id in [0,1,2,3,4,5]:
                turn_ids.append(0)
                # 遇到<del>或者<sep>，表示进入下一轮会话
                if id == 0 or id ==1:
                    turn_id += 1
            else:
                turn_ids.append(turn_id)

        inputs, targets = _encode(sent2, token2idx, maxlen2, "y")

        yield (x,turn_ids, sent1.decode('utf-8')), (inputs, targets, sent2.decode('utf-8'))

def _input_fn(sents1, sents2, vocab_fpath, batch_size, gpu_nums, maxlen1, maxlen2, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([maxlen1+maxlen2], [maxlen1+maxlen2],()),
              ([maxlen2], [maxlen2], ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.string))
    # _generator_fn是一个生成器
    # 每次生成(x, sent1.decode('utf-8')), (inputs, targets, sent2.decode('utf-8'))

    dataset = tf.data.Dataset.from_generator(
        _generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath, maxlen1, maxlen2))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size*gpu_nums)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size*gpu_nums)

    return dataset

def get_batch(fpath, maxlen1, maxlen2, vocab_fpath, batch_size, gpu_nums, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath: source file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2 = _load_data(fpath, maxlen1, maxlen2)
    # sents1 sents2 list of string
    batches = _input_fn(sents1, sents2, vocab_fpath, batch_size, gpu_nums, maxlen1, maxlen2, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size*gpu_nums)
    return batches, num_batches, len(sents1)
