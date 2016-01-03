import pandas as pd

# Kaggle predict cateogry from dataset.
## Load data and preprocessing with making word2vec

train_data = pd.read_csv('./data/train.csv')

test_data = pd.read_csv('./data/test.csv')

train_data = train_data.drop(['X', 'Y'], axis = 1)

train_data['Dates'] = train_data['Dates'].str.split(' ').str[0]
train_data['Dates'] = train_data['Dates'].str.split('-').str[0] + train_data['Dates'].str.split('-').str[1] + train_data['Dates'].str.split('-').str[2]

train_data['com'] = (train_data['Dates'] + ' ' +train_data['Descript'] + ' ' + train_data['DayOfWeek'] + ' '+ train_data['PdDistrict']+ ' ' + train_data['Resolution']+ ' ' + train_data['Address'])

grouped_train_data = train_data.groupby('Category')

sentences = []

for k, v in grouped_train_data:
    sentences += v['com'].str.split().tolist()

import gensim

model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

model.save('./data/word2vec')



