"""
混合前端的seq2seq模型旨在通过结合多种处理技术和数据源，增强输入数据的表示能力，从而提高序列到序列任务的性能。其主要目的是：

多模态处理：结合不同类型的输入数据（如文本、音频、图像），以捕捉更丰富的信息。

特征融合：集成多种特征提取方法（如CNN和RNN），以捕获局部和序列模式。

复杂输入的处理：通过多种前端组件，提升模型对复杂输入的处理能力。

性能提升：通过丰富输入表示，提高机器翻译、语音识别等任务的准确性和鲁棒性。

这种混合前端的设计使得seq2seq模型能够更好地应对多样化的输入场景，提升整体表现。
"""


"""准备工作"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import torch
import torch.nn as nn #networking
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

# 使用cpu
device = torch.device("cpu")

MAX_LENGTH = 10 # Maximum sentence length

# 默认的词向量
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token

"""
Voc类主要用于构建和管理词汇表，其主要功能包括：

1. 初始化词汇表：创建单词到索引、索引到单词的映射，并包含PAD、SOS、EOS三个特殊标记

2. 添加句子：将句子分割成单词并逐个添加到词汇表中

3. 添加单词：如果单词不在词汇表中，则添加新词并更新计数；如果已存在，则增加计数

4. 修剪词汇表：移除出现次数低于指定阈值的低频词，保留常用词

5. 维护词汇表状态：记录词汇表是否已被修剪，统计词汇表大小等

Voc类在自然语言处理任务中起着重要作用，它帮助将文本数据转换为模型可以处理的数字形式，
同时通过修剪功能可以控制词汇表大小，提高模型效率。
"""
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 统计SOS, EOS, PAD

    # 这是一个词汇表类，用于管理单词到索引的映射关系
    # trimmed: 标记是否已经对词汇表进行过修剪（去除低频词）
    # word2index: 字典，存储单词到索引的映射
    # word2count: 字典，存储每个单词出现的次数
    # index2word: 字典，存储索引到单词的映射，初始化时包含三个特殊标记
    # num_words: 词汇表当前大小，初始为3（包含PAD、SOS、EOS三个特殊标记）
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    # 该方法用于修剪词汇表，移除出现次数低于指定阈值的单词
    # min_count: 单词保留的最小出现次数
    # 首先检查是否已经修剪过，如果已经修剪则直接返回
    # 然后标记为已修剪状态
    # 创建一个空列表keep_words用于存储需要保留的单词
    # 遍历word2count字典，保留出现次数大于等于min_count的单词
    # 打印保留单词的比例信息
    # 重新初始化所有字典，只保留PAD、SOS、EOS三个特殊标记
    # 将保留的单词重新添加到词汇表中
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # 统计默认的令牌
        for word in keep_words:
            self.addWord(word)

# 小写并删除非字母字符
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 使用字符串句子，返回单词索引的句子
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]



"""
定义编码器:
编码器RNN模块的作用是将输入的句子序列转换为固定维度的隐藏状态表示。

主要功能：
1. 通过嵌入层将单词索引转换为词向量
2. 使用GRU（门控循环单元）处理序列数据
3. 支持双向GRU，可以同时考虑前后文信息
4. 处理变长序列，使用pack_padded_sequence和pad_packed_sequence来优化计算

参数说明：
- hidden_size: 隐藏层维度大小，决定编码器输出的维度
- embedding: 预定义的嵌入层，用于将单词索引转换为词向量
- n_layers: GRU的层数，默认为1
- dropout: 用于防止过拟合的dropout概率，默认为0

forward方法流程：
1. 将输入的单词索引序列通过嵌入层转换为词向量
2. 使用pack_padded_sequence处理变长序列，提高计算效率
3. 将处理后的序列输入GRU，得到输出和隐藏状态
4. 使用pad_packed_sequence将输出序列重新填充为原始长度
5. 如果是双向GRU，将前向和后向的输出相加
6. 返回最终的输出序列和隐藏状态

这个编码器模块是seq2seq模型的重要组成部分，负责将输入序列编码为固定维度的表示，供解码器使用。
"""

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化GRU;input_size和hidden_size参数都设置为'hidden_size'
        # 因为我们输入的大小是一个有多个特征的词向量== hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 将单词索引转换为向量
        embedded = self.embedding(input_seq)
        # 为RNN模块填充批次序列
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # 正向通过GRU
        outputs, hidden = self.gru(packed, hidden)
        # 打开填充
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 将双向GRU的输出结果总和
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # 返回输出以及最终的隐藏状态
        return outputs, hidden


"""定义解码器的注意力模块"""

"""
注意力机制是seq2seq模型中的重要组成部分，它允许解码器在生成每个词时关注输入序列的不同部分。

主要功能：
1. 计算解码器当前隐藏状态与编码器所有输出之间的相关性（注意力分数）
2. 根据相关性权重对编码器输出进行加权求和，得到上下文向量
3. 帮助模型更好地处理长序列，缓解信息丢失问题

实现方法：
1. 点积注意力（dot）：直接计算隐藏状态和编码器输出的点积
2. 通用注意力（general）：通过一个线性层转换编码器输出后再计算点积
3. 拼接注意力（concat）：将隐藏状态和编码器输出拼接后通过线性层和tanh激活函数处理

参数说明：
- method: 注意力计算方法，支持'dot', 'general', 'concat'
- hidden_size: 隐藏层维度大小，用于初始化线性层参数

forward方法流程：
1. 根据选择的注意力方法计算注意力分数
2. 对注意力分数进行softmax归一化，得到注意力权重
3. 返回归一化后的注意力权重，用于后续的加权求和

"""

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
    #点积注意力
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    #通用注意力
    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    #拼接注意力
    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力权重（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 转置max_length和batch_size维度
        attn_energies = attn_energies.t()

        # 返回softmax归一化概率分数（增加维度）
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

"""
这段代码实现了一个基于注意力机制的序列到序列模型中的解码器部分，具体是Luong注意力机制。

主要组件和功能：

1. 注意力机制(Attn类):
   - 实现了三种注意力计算方法：点积、通用和拼接
   - 根据隐藏状态和编码器输出计算注意力权重，最终输出的是注意力权重
   - 使用softmax对注意力分数进行归一化


2. LuongAttnDecoderRNN类:
   - 继承自nn.Module，是一个标准的PyTorch模块
   - 包含以下主要层：
     * embedding：将输入词索引转换为词向量
     * gru：门控循环单元，用于处理序列数据
     * concat：线性层，用于连接GRU输出和上下文向量
     * out：输出层，生成最终的概率分布
     * attn：注意力机制实例

3. forward方法流程：
   - 将输入词转换为嵌入向量
   - 通过GRU处理得到隐藏状态
   - 计算注意力权重
   - 生成加权上下文向量
   - 连接GRU输出和上下文向量
   - 通过线性层和softmax生成输出概率分布
   - 返回输出和新的隐藏状态

这个解码器在机器翻译等序列到序列任务中非常有用，它能够动态地关注输入序列的不同部分，从而生成更准确的输出。
"""

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 保持参考
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义层
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 注意：我们这步只运行一次
        # 获取当前输入字对应的向量映射
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 通过单向GRU转发
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 通过当前GRU的输出计算注意力权重
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 使用Luong的公式5来连接加权上下文向量和GRU输出
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # 使用Luong的公式6来预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # 返回输出和最终的隐藏状态
        return output, hidden


"""
定义评估：
-贪婪搜索解码器
我们使用GreedySearchDecoder模块来简化实际的解码过程。
该模块将训练好的编码器和解码器模型作为属性， 
驱动输入语句(词索引向量)的编码过程，并一次一个词(词索引)迭代地解码输出响应序列。
"""

class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']



    """
    这段代码实现了一个基于贪婪搜索的解码器，用于序列到序列模型中的推理过程。主要功能如下：

    1. 初始化参数：
       - encoder: 编码器模型，用于将输入序列编码为上下文向量
       - decoder: 解码器模型，用于生成输出序列
       - decoder_n_layers: 解码器的层数
       - _device: 指定模型运行的设备（CPU或GPU）
       - _SOS_token: 序列开始标记，用于初始化解码过程
       - _decoder_n_layers: 解码器的层数

    2. 前向传播过程：
       - 首先将输入序列通过编码器，得到编码器输出和隐藏状态
       - 使用编码器的最后隐藏状态初始化解码器的隐藏状态
       - 使用SOS_token初始化解码器的第一个输入
       - 迭代生成输出序列，每次生成一个token
       - 在每次迭代中：
         * 通过解码器生成当前token的概率分布
         * 选择概率最大的token作为当前输出
         * 将生成的token和对应的概率记录下来
         * 将当前token作为下一个解码步骤的输入
       - 最终返回生成的token序列和对应的概率

    3. 特点：
       - 使用贪婪搜索策略，每次选择概率最大的token
       - 适用于序列生成任务，如机器翻译、文本摘要等
       - 实现简单，计算效率高，但可能无法找到全局最优解

    4. 使用示例：
       - 创建GreedySearchDecoder实例，传入训练好的编码器和解码器
       - 调用forward方法，传入输入序列和最大生成长度
       - 返回生成的token序列和对应的概率
    """
    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # 首先将输入序列通过编码器，得到编码器输出和隐藏状态
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 使用编码器的最后隐藏状态初始化解码器的隐藏状态
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # 使用SOS_token初始化解码器的第一个输入
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

"""
接下来，我们定义一些函数来计算输入。

求值函数`evaluate`接受一个规范化字符串语句，将其处理为其对应的单词索引张量(批处理大小为1)，

并将该张量传递给一个名为`searcher`的`GreedySearchDecoder`实例，以处理编码/解码过程。

检索器返回输出的单词索引向量和一个分数张量，该张量对应于每个解码的单词标记的`softmax`分数。

最后一步是使用`voc.index2word`将每个单词索引转换回其字符串表示形式。

"""


"""
求值函数`evaluate`接受一个规范化字符串语句，将其处理为其对应的单词索引张量(批处理大小为1)
"""
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # 格式化输入句子作为批处理
    # words -> indexes
    # ---接受一个规范化字符串语句，将其处理为其对应的单词索引张量(批处理大小为1)，
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建长度张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置批量的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用适当的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    # ---并将该张量传递给一个名为`searcher`的`GreedySearchDecoder`实例，以处理编码/解码过程。
    # 用searcher解码句子
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words

    # ---最后一步是使用`voc.index2word`将每个单词索引转换回其字符串表示形式。
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

"""
我们还定义了两个函数来计算输入语句。
`evaluateInput`函数提示用户输入，并计算输入。它持续请求另一次输入，直到用户输入“q”或“quit”。
`evaluateExample`函数只接受一个字符串输入语句作为参数，对其进行规范化、计算并输出响应。
"""
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 获取输入的句子
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 规范化句子
            input_sentence = normalizeString(input_sentence)
            # 评估句子
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 格式化和打印回复句
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
# 规范化输入句子并调用evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # 规范化句子
    input_sentence = normalizeString(sentence)
    # 评估句子
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))




"""
加载预训练参数
"""

save_dir = os.path.join("data", "save.pth")
corpus_name = "cornell movie-dialogs corpus"

# 配置模型
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 如果你加载的是自己的模型
# 设置要加载的检查点
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))

# 如果你加载的是托管模型
loadFilename = 'data/4000_checkpoint.tar'

# 加载模型
# 强制CPU设备选项（与本教程中的张量匹配）
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']

encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']

embedding_sd = checkpoint['embedding']

voc = Voc(corpus_name)
voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')

# 初始化词向量
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)

# 初始化编码器和解码器模型
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

# 加载训练模型参数
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# 使用适当的设备
encoder = encoder.to(device)
decoder = decoder.to(device)
# 将dropout层设置为eval模式
encoder.eval()
decoder.eval()
print('Models built and ready to go!')


"""
将模型转化为Torch脚本
"""
### 转换编码器模型
# 创建人工输入
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words).to(device)
test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)
# 跟踪模型
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))


### 转换解码器模型
# 创建并生成人工输入
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# 跟踪模型
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))


### 初始化searcher模块
scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)



"""
图形打印
"""
print('encoder graph', traced_encoder.__getattr__('forward').graph)
print('decoder graph', traced_decoder.__getattr__('forward').graph)




"""
运行结果评估
"""
# 评估例子
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, traced_encoder, traced_decoder, scripted_searcher, voc)



"""
保存模型
"""

scripted_searcher.save(save_dir)
