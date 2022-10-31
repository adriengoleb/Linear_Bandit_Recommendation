# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:16:03 2022

# Using epsilon greeding offline learning method
# To be done(maybe ? depends the time we have) :
    I removed the slate_size(the number of recommendations to make at each step) and the 
    batch_size(number of user sessions to observe for each iteration of the bandit) param to simplify the pb.
    We can test those params later
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#we focus on the userId = 414 who gave 2698 ratings starting from timestamp 961436216 

def UCB_strategy(df, t, rewards_estimation, alpha = 2) :
    nbMovie = len(rewards_estimation)
    choice = rewards_estimation.iloc[0]['movieId']
    max_ucb = rewards_estimation.iloc[0]['mean_rating'] + alpha * (np.sqrt(np.log(t) / rewards_estimation.iloc[0]['nb_rating']))

    for i in range(1, nbMovie): 
        ucb_i = rewards_estimation.iloc[i]['mean_rating'] +  (np.sqrt((np.log(t) * alpha)  / rewards_estimation.iloc[i]['nb_rating']))
        if(ucb_i>max_ucb) :
            choice = rewards_estimation.iloc[i]['movieId']
            max_ucb = ucb_i
      
    return choice


#update rewards estimation after each movie choice if the recommendation matches logged data, ignore otherwise
def update_rewards_estimation(df, rewards, rewards_estimation, t, choice, user) :  
        user_rating = df[df['movieId'] ==  choice].rating.item()
        rewards.append(user_rating)
        if rewards_estimation['movieId'].isin([choice]).any():
            mean_rating = rewards_estimation[rewards_estimation.movieId == choice].mean_rating.item()
            nb_rating = rewards_estimation[rewards_estimation.movieId == choice].nb_rating.item() + 1
            mean_rating = (mean_rating * (nb_rating-1) + user_rating)/nb_rating
            rewards_estimation.loc[rewards_estimation['movieId'] == choice, 'mean_rating'] = mean_rating
            rewards_estimation.loc[rewards_estimation['movieId'] == choice, 'nb_rating'] = nb_rating
            rewards_estimation.loc[rewards_estimation['movieId'] == choice, 'last_update_t'] = t
        else:
            rewards_estimation.loc[len(rewards_estimation)] = [choice, user_rating, 1, t]

        return rewards, rewards_estimation


def test_alpha(df, alphas, N=1000):
    for alpha in alphas :
        rewards = []
        rewards_estimation = pd.DataFrame(data=None, columns=['movieId','mean_rating', 'nb_rating', 'last_update_t'])
        #init the rewards_estimation
        nbMovie = len(df)
        for i in range(nbMovie):
            movieId = df.iloc[i]['movieId']
            rating = random.randint(0,5)
            new_row = {'movieId':movieId, 'mean_rating':rating, 'nb_rating':1, 'last_update_t':0}
            #append row to the dataframe
            rewards_estimation = rewards_estimation.append(new_row, ignore_index=True)
        #t is the time step
        for t in range(N):
            current_user = df.userId.iloc[0]
            choice = UCB_strategy(df,t, rewards_estimation, alpha)
            rewards, rewards_estimation = update_rewards_estimation(df, rewards, rewards_estimation, t, choice, current_user)
            print(rewards_estimation)
        
        regrets = [5-reward for reward in rewards]
        cummulative_regrets= []
        for i in range(len(regrets)):
            cummulative_regrets.append(sum(regrets[0:i+1]))
        label = 'alpha '+ str(alpha)
        plt.plot(cummulative_regrets, label = label)
        plt.legend()

    


### Modelisation starts from here :  
ratings = pd.read_csv('ratings.csv')
#Remove movies with < 2000 ratings and user who gave less than 1000 ratings
movies_to_keep = pd.DataFrame(ratings.movieId.value_counts()).loc[pd.DataFrame(ratings.movieId.value_counts())['movieId']>=2000].index
df = ratings[ratings.movieId.isin(movies_to_keep)]
users_to_keep = pd.DataFrame(df.userId.value_counts()).loc[pd.DataFrame(df.userId.value_counts())['userId']>=1000].index
df = df[df.userId.isin(users_to_keep)]
nbUser = df.userId.value_counts().shape[0]
nbMovie = df.movieId.value_counts().shape[0]
user = df.sample().userId.values[0]
movieList = df[df['userId']==user].movieId.tolist()
df = df.loc[df['movieId'].isin(movieList)]
#keep a record of the ratings given by the user
user_rating = df[df.userId == user]
test_alpha(user_rating,[2,3,5],N=3000)
