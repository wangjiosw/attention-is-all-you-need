import spacy

import pandas as pd
from sklearn.model_selection import train_test_split
import config
from seq2seq_data import seq2seqData



def loadData():
    """
    load data and transform data to '.csv' format
    after run it , got train.csv, val.csv and test.csv
    """
    europarl_en = open('data/europarl-v7.fr-en.en', encoding='utf-8').read().split('\n')
    europarl_fr = open('data/europarl-v7.fr-en.fr', encoding='utf-8').read().split('\n')

    europarl_en = europarl_en[:50000]
    europarl_fr = europarl_fr[:50000]

    # 也许与直觉相反，使用Torchtext的最佳方法是将数据转换为电子表格格式，无论数据文件的原始格式如何。
    raw_data = {'English': europarl_en, 'French': europarl_fr}
    df = pd.DataFrame(raw_data, columns=["English", "French"])

    # remove very long sentences and sentences where translations are not of roughly equal length
    df['eng_len'] = df['English'].str.count(' ')
    df['fr_len'] = df['French'].str.count(' ')
    df = df.query('fr_len < 80 & eng_len < 80')
    df = df.query('fr_len < eng_len * 1.5 & fr_len * 1.5 > eng_len')

    df = df.drop(['eng_len', 'fr_len'], 1)

    # We now have to make a validation set.
    # create train and validation set
    train, val = train_test_split(df, test_size=0.1)
    train, test = train_test_split(train, test_size=0.2)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    test.to_csv("test.csv", index=False)
    return list(train.columns)


en = spacy.load('en')
fr = spacy.load('fr')


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


def tokenize_fr(sentence):
    return [tok.text for tok in fr.tokenizer(sentence)]





if __name__ == "__main__":

    cols = loadData()
    translateDataset = seq2seqData(tokenize_en, tokenize_fr, cols, config.BATCH_SIZE, config.DEVICE)

    train_iter, val_iter, test_iter = translateDataset.generateIterator()

    batch = next(iter(train_iter))
    out = batch.English
    print(out.shape)
    print(out)
















