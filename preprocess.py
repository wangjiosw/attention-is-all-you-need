import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext import data, datasets
import config
from torch.nn import init



def loadData():
    europarl_en = open('data/europarl-v7.fr-en.en', encoding='utf-8').read().split('\n')
    europarl_fr = open('data/europarl-v7.fr-en.fr', encoding='utf-8').read().split('\n')

    europarl_en = europarl_en[:10000]
    europarl_fr = europarl_fr[:10000]

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
    train, val = train_test_split(df, test_size=0.2)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    return


en = spacy.load('en')
fr = spacy.load('fr')


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


def tokenize_fr(sentence):
    return [tok.text for tok in fr.tokenizer(sentence)]


# define Field
EN_TEXT = Field(tokenize=tokenize_en)
FR_TEXT = Field(tokenize=tokenize_fr, init_token="<sos>", eos_token="<eos>")

# associate the text in the 'English' column with the EN_TEXT field,
# and 'French' with FR_TEXT
loadData()
data_fields = [('English', EN_TEXT), ('French', FR_TEXT)]
train, val = data.TabularDataset.splits(path='./', train='train.csv',
                                        validation='val.csv', skip_header=True, format='csv', fields=data_fields)

# build vocab
EN_TEXT.build_vocab(train, val)  # vectors='glove.6B.100d', max_size=30000)
# EN_TEXT.vocab.vectors.unk_init = init.xavier_uniform

FR_TEXT.build_vocab(train, val)  # vectors='glove.6B.100d', max_size=30000)
# FR_TEXT.vocab.vectors.unk_init = init.xavier_uniform

# generate iterator
train_iter = data.BucketIterator(train, batch_size=config.BATCH_SIZE,
                                 sort_key=lambda x: len(x.French), shuffle=True, device=config.DEVICE)
val_iter = data.BucketIterator(val, batch_size=config.BATCH_SIZE,
                               sort_key=lambda x: len(x.French), shuffle=True, device=config.DEVICE)

if __name__ == "__main__":
    print('info:')
    print('len(EN_TEXT.vocab): %d \t len(FR_TEXT.vocab) %d'%(len(EN_TEXT.vocab), len(FR_TEXT.vocab)))
    print('save dataset')


