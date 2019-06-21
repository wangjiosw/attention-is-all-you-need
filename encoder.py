from attention import MultiHeadedSelfAttention
from LayerNorm import LayerNorm
from FFN import FFN
import torch
import torch.nn as nn
import torch.nn.functional as F
from pre import tokenize_en, tokenize_fr, seq2seqData
import config


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        self.multi_attention = MultiHeadedSelfAttention()
        self.layerNorm1 = nn.LayerNorm(input_shape[1:])
        self.layerNorm2 = nn.LayerNorm(input_shape[1:])
        self.FNN = FFN()
        self.dropout = nn.Dropout(config.P_drop)

    def forward(self, x):
        """
        :param x: (batch_size, n, 512)
        :return: (batch_size, n, 512)
        """
        z = self.multi_attention(x)
        z = self.layerNorm1(x+z)
        z = self.dropout(z)

        out = self.FNN(z)
        out = self.layerNorm2(z+out)
        out = self.dropout(out)

        return out


if __name__ == "__main__":
    cols = ['English', 'French']
    translateDataset = seq2seqData(tokenize_en, tokenize_fr, cols, config.BATCH_SIZE, config.DEVICE, data_path='data')

    train_iter, val_iter, test_iter = translateDataset.generateIterator()

    batch = next(iter(train_iter))

    in_vocab, _ = translateDataset.getVocab()
    len_in_vocab = len(in_vocab)

    data = batch.English
    embed = nn.Embedding(len_in_vocab, 512)
    embed.to(config.DEVICE)
    data = embed(data)
    print(data.shape, type(data))

    model = Encoder(data.shape)
    model.to(config.DEVICE)
    out = model(data)

    print(model)
    print(out.shape)
