from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from ctt.clean import kitchen_sink

VAL_RATIO = 0.1
TEST_RATIO = 0.2
MIN_NUM_TOKS = 2
NUMBER_PAIRS = 100000
MAX_VOCAB = 20000

class PairwiseData():
    def __init__(self):
        df = pd.read_csv('../data/imdb_movie_reviews/labeledMovieData.csv',sep=',')        
        df['text'] = df['text'].apply(lambda x: kitchen_sink(x))
        # split into train/val/test
        train, test = train_test_split(df, test_size = VAL_RATIO + TEST_RATIO)
        test, val = train_test_split(test, test_size = VAL_RATIO/(VAL_RATIO + TEST_RATIO))  
        # create and train vectorizer
        vectorizer = CountVectorizer(min_df=20, max_df=0.7, max_features=MAX_VOCAB)
        self.vectorizer = vectorizer.fit(train['text'].astype(str))
        # set vocab size now that we know it
        self.vocab_size = len(vectorizer.vocabulary_)
        # get bow vectors
        self.bows = vectorizer.transform(df.iloc[:,0].astype(str)).astype(np.int16)
        # get bow for datasets
        bows_train, bows_val, bows_test = [
            vectorizer.transform(d.iloc[:,0].astype(str)).astype(np.int16)
            for d in [train, val, test]
        ]
        # find document indices where number of tokens is >=min_num_toks
        ix_keep_train, ix_keep_val, ix_keep_test = [
            np.argwhere(np.asarray(bow.sum(axis=-1)).squeeze() >= MIN_NUM_TOKS).squeeze()
            for bow in [bows_train, bows_val, bows_test]
        ]
        # subset bows to these indices
        # NOTE: these are not currently used
        self.bows_train = bows_train[ix_keep_train]
        self.bows_val = bows_val[ix_keep_val]
        self.bows_test = bows_test[ix_keep_test]
        # subset dfs for later lookups
        self.train = train.iloc[ix_keep_train]
        self.val = val.iloc[ix_keep_val]
        self.test = test.iloc[ix_keep_test]
        
        
    def get_pairs_table(self, df):
        pair_ix = np.random.choice(df.index.values, size=(NUMBER_PAIRS,2), replace=True)
        starsA = df.stars_category[pair_ix[:,0]]
        starsB = df.stars_category[pair_ix[:,1]]

        is_similar = (starsA.values == starsB.values)

        return pd.DataFrame({'ix_A': pair_ix[:,0],
                             'ix_B': pair_ix[:,1],
                             'label': is_similar})
    
    

class DocumentPairData(Dataset):

    def __init__(self, bows, index_table, prob=0.5):
        """
        Integrated dataset loader (supervised and unsupervised dataset)
        
        Args:
            data: an pre-loaded sparse matrix
            index_table: dataframe/list of pre-split pairwise indices, where index_table[i] = [a_ix, b_ix, label]
            prob (float): between [0,1], the probability of returning a supervised sample

        Return:
            {'a': bow, 'b': bow, 'label': bool, 'observed': bool}
        """
        self.bows = bows
        self.index_table = index_table
        self.prob = prob
            
    def __len__(self):
        return len(self.index_table)

    def __getitem__(self, ix): 
        
        a_ix = self.index_table.iloc[ix][0]
        b_ix = self.index_table.iloc[ix][1]
        label = self.index_table.iloc[ix][2]
                
        a = self.bows[a_ix].toarray().astype(np.float32)
        b = self.bows[b_ix].toarray().astype(np.float32)
        
        is_observed = (np.random.rand() < self.prob)
        
        return {'a': a, 'b': b, 'label': label, 'observed': is_observed}

    