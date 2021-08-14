import pickle
import tqdm
F=open(r'vocab.pkl','rb')
vocab=pickle.load(F)
# print(content)
# print(len(content))

def biGramHash(sequence, t, buckets):  # words_line, i, buckets  len(words_line) = 32  buckets=250499
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets

def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

buckets = 250499
tokenizer = lambda x: [y for y in x]
pad_size = 32
UNK, PAD = '<UNK>', '<PAD>'
contents = []
with open("train.txt", 'r', encoding='UTF-8') as f:
    for line in f:
        lin = line.strip()
        if not lin:
            continue
        content, label = lin.split('\t')
        words_line = []
        # print("content:", content)
        # print("label:",label)
        token = tokenizer(content)
        seq_len = len(token)
        # print("token:", token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        # print(words_line)
        bigram = []
        trigram = []
        for i in range(pad_size):  # pad_size: 32
            bigram.append(biGramHash(words_line, i, buckets))  # len(words_line) = 32  buckets=250499
            trigram.append(triGramHash(words_line, i, buckets))
        # print(bigram)
        # print(trigram)
        contents.append((words_line, int(label), seq_len, bigram, trigram))
print(contents)
# t = ['连', '续', '上', '涨', '7', '个', '月', ' ', '糖', '价', '7', '月']
# t.extend([PAD] * 4)
#
# print(t)

