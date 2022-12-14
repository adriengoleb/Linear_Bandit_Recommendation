# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:09:23 2022

@author: sihan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:10:53 2022

@author: sihan
"""

import pandas as pd
import numpy as np 
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import time

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
#Remove movies with < 2000 ratings and user who gave less than 1000 ratings
movies_to_keep = pd.DataFrame(ratings.movieId.value_counts()).loc[pd.DataFrame(ratings.movieId.value_counts())['movieId']>=2000].index
df = ratings[ratings.movieId.isin(movies_to_keep)]
users_to_keep = pd.DataFrame(df.userId.value_counts()).loc[pd.DataFrame(df.userId.value_counts())['userId']>=1000].index
df = df[df.userId.isin(users_to_keep)]
nbUser = df.userId.value_counts().shape[0]
nbMovie = df.movieId.value_counts().shape[0]

user = df.sample().userId.values[0] #61836
movieList = df[df['userId']==user].movieId.tolist()
df = df.loc[df['movieId'].isin(movieList)]
#keep a record of the ratings given by the user
user_rating = df[df.userId == user]
#we suppose that this user joined in and had no ratings 
#df.drop(df[df['userId'] == user ].index, inplace = True)
R = df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
M = R.to_numpy()


#Construct the feature vector
#Decompose the matrix M to user features matrix and movie feature matrix
#for now I chose 30 but it would be better to use cross validation to choose the number of the features : https://towardsdatascience.com/how-to-use-cross-validation-for-matrix-completion-2b14103d2c4c
#Method 1 : using rating to generate the feature vector of film
from sklearn.decomposition import NMF
model = NMF(n_components=10, init='random', random_state=0, max_iter=5000)
W = model.fit_transform(M) #feauture user
H = model.components_ #feature movie
err = np.linalg.norm(M - W @ H)**2/np.linalg.norm(M)**2 
users_feature = W
movies_feature = H.T


#Method 2 : using the movie category to generate de feature vector of film
movies = movies.join(movies.genres.str.get_dummies().astype(int))
movies.drop(['genres','title'], inplace=True, axis=1)
movies_feature_genre = pd.merge(user_rating, movies, how='inner', left_on = 'movieId', right_on = 'movieId')
movies_feature_genre.drop(['userId', 'movieId', 'rating', 'timestamp'], inplace=True, axis=1)
movies_feature_genre = movies_feature_genre.values.tolist()
movies_feature_genre = np.array(movies_feature_genre)

#Method 3 : apply PCA to the method to get a low-dimension feature vector
from sklearn import decomposition
pca = decomposition.PCA(n_components=10)
movies_feature_genre_pca = pca.fit_transform(movies_feature_genre)


def LinUCB (movies_feature, movieList, user_rating, alpha, T=1000):
    # d : dimension of the feature vector
    d = movies_feature.shape[1] 
    # n : number of the movie that we can propose
    n = movies_feature.shape[0]
    
    # init
    A = np.repeat(np.identity(d, dtype=float)[np.newaxis, :, :], n, axis=0)
    b = np.zeros(shape=(n, d))
    theta_a = np.zeros((n, d))
    p_t = np.zeros(n)
    #new_A =  [True for i in range(n)]
    reward = []
    regret = []
    
    
    # propose a film which maximise the estimated payoff to the user at each timestep
    for t in range(T):
        # iterate over all movies to init the reward
        for a in range(n):  
            x_ta = movies_feature[a] 
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_ta) + alpha * np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))
        max_p_t = np.max(p_t)
        print(max_p_t)
        a_t = np.random.choice(np.argwhere(p_t == max_p_t).flatten())  # with ties broken arbitrarily to decide the movie to recommend to user
        print(a_t)
        x_t_at = movies_feature[a_t]
        #observe a real-valued reward at time t
        rating = user_rating.iloc[[a_t]].rating.values[0]
        reward.append(rating)
        regret.append(user_rating.rating.max()-rating)
        A[a_t] = A[a_t] + np.outer(x_t_at, x_t_at)
        b[a_t] = b[a_t] + rating * x_t_at

    return reward, regret



#Test LinUCB
T = 3000
alpha = 2.5
#case 1 : movies_feature
start_time = time.time()
reward, regret = LinUCB(movies_feature, movieList, user_rating, alpha,  T) 
total_time = time.time() - start_time#337.953284740448

#Calculate the cumulative regret and reward
cum_reward = []
cum_regret = []
for i in range(len(reward)):
  cum_reward.append(sum(reward[0:i+1]))
for i in range(len(regret)):
  cum_regret.append(sum(regret[0:i+1]))

### Plots
label = 'feature vector generated by rating'
plt.plot(cum_regret, label = label)
#plt.xlim(0,T)
#plt.xlabel("Step")
#plt.ylim(0)
#plt.ylabel("Regret")
#plt.title("LinUCB with different feature vector : Evolution of the regret over time")
#plt.show()


#case 2 : movies_feature_genre
alpha = 2.5
start_time = time.time()
reward_genre, regret_genre = LinUCB(movies_feature_genre, movieList, user_rating, alpha,  T) 
total_time_genre = time.time() - start_time#2301.7663309574127


#Calculate the cumulative regret and reward
cum_reward_genre = []
cum_regret_genre = []
for i in range(len(reward_genre)):
  cum_reward_genre.append(sum(reward_genre[0:i+1]))
for i in range(len(regret_genre)):
  cum_regret_genre.append(sum(regret_genre[0:i+1]))

### Plots
label = 'feature vector generated by genre of the movie'
plt.plot(cum_regret_genre, label = label)
#plt.xlim(0,T)
#plt.xlabel("Step")
#plt.ylim(0)
#plt.ylabel("Regret")
#plt.title("LinUCB : Evolution of the regret over time")
#plt.show()

#case 3 : movies_feature_genre_pca
alpha = 2.5
start_time = time.time()
reward_genre_pca, regret_genre_pca = LinUCB(movies_feature_genre_pca, movieList, user_rating, alpha,  T) 
total_time_genre_pca = time.time() - start_time#457.22877621650696


#Calculate the cumulative regret and reward
cum_reward_genre_pca = []
cum_regret_genre_pca = []
for i in range(len(reward_genre_pca)):
  cum_reward_genre_pca.append(sum(reward_genre_pca[0:i+1]))
for i in range(len(regret_genre_pca)):
  cum_regret_genre_pca.append(sum(regret_genre_pca[0:i+1]))

### Plots
label = 'feature vector generated by genre of the movie after applying the PCA'
plt.plot(cum_regret_genre_pca, label = label)
plt.xlim(0,T)
plt.xlabel("Step")
plt.ylim(0)
plt.ylabel("Regret")
plt.title("LinUCB : Evolution of the regret over time (alpha = 2.5)")
plt.show()

#test alpha
def test_alpha(df, movieList, user_rating, alphas, T=1000):
    for alpha in alphas :
        reward_alpha, regret_alpha = LinUCB(df, movieList, user_rating, alpha,  T) 
        cum_reward_alpha = []
        cum_regret_alpha = []
        for i in range(len(reward_alpha)):
            cum_reward_alpha.append(sum(reward_alpha[0:i+1]))
        for i in range(len(regret_alpha)):
            cum_regret_alpha.append(sum(regret_alpha[0:i+1]))

        label = 'alpha '+ str(alpha)
        plt.plot(cum_regret_alpha, label = label)
        plt.legend()
    plt.xlim(0,T)
    plt.xlabel("Step")
    plt.ylim(0)
    plt.ylabel("Regret")
    plt.title("LinUCB with different alpha ")
    plt.show()

test_alpha(movies_feature, movieList, user_rating, [2, 2.5, 3, 5], T=3000)