"""
Discrete Variable Summary Statistics:
- Frequencies of classes
- Mode: Most frequently occuring class
- Quantiles: categories that occur more than t times
- Simple matching for how often two variables are the same, or jacard for binary variables

Continuous Summary Statistics:
- Mean, Median, Quantiles (Measures of Location) --> mean + std dev are more sensitive to outliers
- Range, Variance, Interquantile Ranges (Measures of spread)
- Correlation, rank correlation, euclidean distance, cosine similarity (measures between continuous variables)

Plots:
- One feature as function of the other
- Histogram
- Boxplots
- Scatterplots (and arrays)
- Map coloring + contour plots
- Stream graph
- Treemaps

"""
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(42)
random.seed(42)


class DataExploration:
    def __init__(self):
        return

    @staticmethod
    def prepare_twitter_df():
        # data frames columns
        user_cols = ['user_id', 'created_at', 'collected_at', 'num_following', 'num_followers',
                     'num_tweets', 'len_of_screen_name', 'len_of_profile_description']
        following_cols = ['user_id', 'series_of_num_following']
        tweets_cols = ['user_id', 'tweet_id', 'tweet', 'created_at']

        # content polluters (spam users)
        content_polluters_df = pd.read_csv('./data/social_honeypot/content_polluters.txt',
                                           sep='\t', names=user_cols, header=None).set_index('user_id')
        content_polluters_following_df = pd.read_csv('./data/social_honeypot/content_polluters_followings.txt',
                                                     sep='\t', names=following_cols, header=None).set_index('user_id')
        content_tweets_df = pd.read_csv('./data/social_honeypot/content_polluters_tweets.txt',
                                        sep='\t', names=tweets_cols, header=None).set_index('user_id')

        # add feature of list of following
        content_polluters_df = content_polluters_df.join(content_polluters_following_df, on='user_id', how='inner')
        content_polluters_df['num_following_changes'] = content_polluters_df.series_of_num_following.apply(
            lambda x: len(x))  # todo this seems wrong..
        # TODO: look to add feature as num of following who are spam

        # add feature for counts of tweets per user
        # content_polluters_df = content_polluters_df.join(
        #     content_tweets_df.groupby('user_id')['tweet_id'].count(), on='user_id', how='inner').rename(
        #     columns={'tweet_id': 'num_tweets'})
        # TODO: utilize other features from this data set besides count of tweets

        # add label
        content_polluters_df['is_spam'] = 1

        print(content_polluters_df.shape)

        # real users
        real_users_df = pd.read_csv('./data/social_honeypot/legitimate_users.txt',
                                    sep='\t', names=user_cols, header=None).set_index('user_id')
        real_users_following_df = pd.read_csv('./data/social_honeypot/legitimate_users_followings.txt',
                                              sep='\t', names=following_cols, header=None).set_index('user_id')
        real_users_tweets_df = pd.read_csv('./data/social_honeypot/legitimate_users_tweets.txt',
                                           sep='\t', names=tweets_cols, header=None).set_index('user_id')

        # add feature of list of following
        real_users_df = real_users_df.join(real_users_following_df, on='user_id', how='inner')
        real_users_df['num_following_changes'] = real_users_df.series_of_num_following.apply(
            lambda x: len(x))
        # TODO: look to add feature as num of following who are spam

        # add feature for counts of tweets per user
        # real_users_df = real_users_df.join(
        #     real_users_tweets_df.groupby('user_id')['tweet_id'].count(), on='user_id', how='inner').rename(
        #     columns={'tweet_id': 'num_tweets'})
        # TODO: utilize other features from this data set besides count of tweets

        # add label
        real_users_df['is_spam'] = -1

        print(real_users_df.shape)

        twitter_df = pd.concat([content_polluters_df, real_users_df])

        print(twitter_df.shape)

        return twitter_df

    @staticmethod
    def prepare_speed_dating_df():
        speed_dating_df = pd.read_csv('./data/speeddating.csv', low_memory=False)
        cols_of_interest = ['gender', 'age', 'age_o', 'race', 'race_o', 'importance_same_race',
                            'importance_same_religion', 'field', 'attractive_important', 'sincere_important',
                            'intellicence_important', 'funny_important', 'ambtition_important',
                            'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition',
                            'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner',
                            'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining',
                            'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies',
                            'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'decision']
        todo_see_if_improves = ['d_age', 'samerace', 'pref_o_attractive', 'pref_o_sinsere', 'pref_o_intelligence',
                                'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_o',
                                'sincere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'shared_interests_o',
                                'decision_o', 'match']
        speed_dating_df = speed_dating_df[cols_of_interest]
        print(speed_dating_df.shape)

        for col in cols_of_interest:
            speed_dating_df = speed_dating_df[speed_dating_df[col] != "?"]


        speed_dating_df = speed_dating_df[~speed_dating_df.isin([np.nan, np.inf, -np.inf]).any(1)]
        speed_dating_df = speed_dating_df.dropna(axis=0)

        print('removing empty values shape:')
        print(speed_dating_df.shape)

        return speed_dating_df

    @staticmethod
    def get_hot_encoded_speed_dating(speed_dating_df):
        labelled_speed_dating_df = speed_dating_df.copy()
        le = preprocessing.LabelEncoder()
        ohe = preprocessing.OneHotEncoder()
        cols_to_encode = ['gender', 'race', 'race_o', 'field']
        for col in cols_to_encode:
            labelled_speed_dating_df[col] = le.fit_transform(labelled_speed_dating_df[col])

        encoded_speed_dating_df = labelled_speed_dating_df

        for col in cols_to_encode:
            X = ohe.fit_transform(X=labelled_speed_dating_df[col].values.reshape(-1, 1)).toarray()
            temp_df = pd.DataFrame(X, columns=[col + str(int(i)) for i in range(X.shape[1])])
            encoded_speed_dating_df = pd.concat([encoded_speed_dating_df, temp_df], axis=1)
            encoded_speed_dating_df = encoded_speed_dating_df.drop(col, axis=1)

        for col in encoded_speed_dating_df.columns.values:
            encoded_speed_dating_df[col] = pd.to_numeric(encoded_speed_dating_df[col])

        # print(encoded_speed_dating_df[encoded_speed_dating_df.isin([np.nan, np.inf, -np.inf]).any(1)])
        encoded_speed_dating_df = encoded_speed_dating_df[
            ~encoded_speed_dating_df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64).dropna(axis=0)

        # print(encoded_speed_dating_df)
        #
        # print(np.any(np.isnan(encoded_speed_dating_df)))
        # print(np.all(np.isfinite(encoded_speed_dating_df)))
        # print(encoded_speed_dating_df.shape)

        return encoded_speed_dating_df #, labelled_speed_dating_df, le

    @staticmethod
    def get_train_test_validation(general_df, dataset_name):
        general_df = shuffle(general_df)
        if dataset_name is 'twitter':
            X = general_df.drop('is_spam', axis=1).drop('created_at', axis=1).drop('collected_at', axis=1).drop(
                'series_of_num_following', axis=1)
            y = general_df['is_spam']
        elif dataset_name is 'speed_dating':
            X = general_df.drop('decision', axis=1)
            y = general_df['decision']
        else:
            print('non-valid dataset chosen')
            exit()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train_nn, X_validation_nn, y_train_nn, y_validation_nn = train_test_split(X_train, y_train,
                                                                                    test_size=0.125)  # size = 0.1 of orginal

        # print(X_train)
        # print(twitter_df.head())
        return X_train, X_test, y_train, y_test, X_train_nn, X_validation_nn, y_train_nn, y_validation_nn
