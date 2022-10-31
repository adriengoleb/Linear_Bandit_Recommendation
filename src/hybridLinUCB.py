# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:32:59 2022

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
index_user = 0
userList = df.userId.unique()
for i in range(len(userList)):
    if userList[i] == user:
        index_user = i
        break
    
movieList = df[df['userId']==user].movieId.tolist()
df = df.loc[df['movieId'].isin(movieList)]
#keep a record of the ratings given by the user
user_rating = df[df.userId == user]
#we suppose that this user joined in and had no ratings 
#df.drop(df[df['userId'] == user ].index, inplace = True)
R = df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
M = R.to_numpy()



#Construct the feature vector(z_a and x_a)
#Decompose the matrix M to user features matrix and movie feature matrix
#for now I chose 30 but it would be better to use cross validation to choose the number of the features : https://towardsdatascience.com/how-to-use-cross-validation-for-matrix-completion-2b14103d2c4c
#z_a
from sklearn.decomposition import NMF
model = NMF(n_components=10, init='random', random_state=0, max_iter=5000)
W = model.fit_transform(M) #feauture user
H = model.components_ #feature movie
err = np.linalg.norm(M - W @ H)**2/np.linalg.norm(M)**2 
users_feature = W
movies_feature = H.T
user_feature = W[index_user]



#Hybrid LinUCB
def hybridLinUCB(movies_feature, user_feature, movieList, user_rating, alpha, T=500):
    # d : dimension of the feature vector
    d = movies_feature.shape[1] 
    # n : number of the movie that we can propose
    n = movies_feature.shape[0]
    k = d*d
    
    # init
    #Building blocks for shared features
    A0 = np.identity(k)
    b0 = np.zeros((k, 1))
    #Building blocks for each arm
    A = np.repeat(np.identity(d, dtype=float)[np.newaxis, :, :], n, axis=0)
    B = np.repeat(np.zeros(shape=(d, k))[np.newaxis, :, :], n, axis=0)
    b = np.zeros(shape=(n, d))
    #theta_a = np.zeros((n, d))
    p_t = np.zeros(n)
    reward = []
    regret = []
    
    for t in range(T):
        A0_inv = np.linalg.inv(A0)
        beta = A0_inv.dot(b0) #Shared coefficient at each time step
        for a in range(n):  
            x_ta = movies_feature[a]
            z_ta = np.outer(movies_feature[a],user_feature).flatten()
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot((b[a]-B[a].dot(beta).T).T) #Coefficient for each arm
            s_ta = z_ta.dot(A0_inv).dot(z_ta) - 2*z_ta.dot(A0_inv).dot(B[a].T).dot(A_a_inv).dot(x_ta) + x_ta.dot(A_a_inv).dot(x_ta) + x_ta.dot(A_a_inv).dot(B[a]).dot(A0_inv).dot(B[a].T).dot(A_a_inv).dot(x_ta)
            p_t[a] = (beta.T.dot(z_ta) + x_ta.dot(theta_a) + alpha * np.sqrt(s_ta))[0]
        
        #Choose movie
        max_p_t = np.max(p_t)
        print(max_p_t)
        a_t = np.random.choice(np.argwhere(p_t == max_p_t).flatten())  # with ties broken arbitrarily to decide the movie to recommend to user
        print(a_t)
        x_t_at = movies_feature[a_t]     
        z_t_at =  np.outer(x_t_at, user_feature).flatten()
        A_at_inv = np.linalg.inv(A[a_t])
        
        #Update global rewards
        A0 = A0 + B[a_t].T.dot(A_at_inv).dot(B[a_t])
        b0 = (b0.T + B[a_t].T.dot(A_at_inv).dot(b[a_t])).T
        #Update local rewards
        rating = user_rating.iloc[[a_t]].rating.values[0]
        reward.append(rating)
        regret.append(user_rating.rating.max()-rating)
        A[a_t] = A[a_t] +  np.outer(x_t_at, x_t_at)
        B[a_t] = B[a_t] +  np.outer(x_t_at, z_t_at)
        b[a_t] = b[a_t] + rating * x_t_at 
        #Update shared features
        A0 = A0 + np.outer(z_t_at,z_t_at) - (B[a_t].T).dot(A_at_inv).dot(B[a_t])
        b0 = (b0.T + rating * z_t_at - (B[a_t].T).dot(A_at_inv).dot(b[a_t])).T
    return reward, regret


#Hybrid LinUCB (Prohibit to choose the same movie)
def hybridLinUCB_bis (movies_feature, user_feature, movieList, user_rating, alpha, T=1000):
    # d : dimension of the feature vector
    d = movies_feature.shape[1] 
    # n : number of the movie that we can propose
    n = movies_feature.shape[0]
    k = d*d
    
    # init
    #Building blocks for shared features
    A0 = np.identity(k)
    b0 = np.zeros((k, 1))
    #Building blocks for each arm
    A = np.repeat(np.identity(d, dtype=float)[np.newaxis, :, :], n, axis=0)
    B = np.repeat(np.zeros(shape=(d, k))[np.newaxis, :, :], n, axis=0)
    b = np.zeros(shape=(n, d))
    #theta_a = np.zeros((n, d))
    p_t = np.zeros(n)
    reward = []
    regret = []
    non_select = dict.fromkeys(movieList, True)
    
    for t in range(T):
        A0_inv = np.linalg.inv(A0)
        beta = A0_inv.dot(b0) #Shared coefficient at each time step
        for a in range(n):  
            x_ta = movies_feature[a]
            z_ta = np.outer(movies_feature[a],user_feature).flatten()
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot((b[a]-B[a].dot(beta).T).T) #Coefficient for each arm
            s_ta = z_ta.dot(A0_inv).dot(z_ta) - 2*z_ta.dot(A0_inv).dot(B[a].T).dot(A_a_inv).dot(x_ta) + x_ta.dot(A_a_inv).dot(x_ta) + x_ta.dot(A_a_inv).dot(B[a]).dot(A0_inv).dot(B[a].T).dot(A_a_inv).dot(x_ta)
            if s_ta<0 :
                s_ta = 0
            p_t[a] = (beta.T.dot(z_ta) + x_ta.dot(theta_a) + alpha * np.sqrt(s_ta))[0]
    
        
        #Choose movie
        choose = False
        a_t = 0
        p_t_temp = p_t
        p_t_temp = np.sort(p_t_temp)
        p_t_temp = p_t_temp[::-1]
        i = 0
        while (not choose) and (i < n) : 
            max_p_t = p_t_temp[0]
            a_t = np.random.choice(np.argwhere(p_t == max_p_t).flatten())  # with ties broken arbitrarily to decide the movie to recommend to user    
            if non_select[movieList[a_t]] : 
                choose = True
            else :
                i = i+1
                p_t_temp = p_t_temp.tolist()
                p_t_temp.remove(p_t_temp[0])
                p_t_temp = np.array(p_t_temp)
        
        #if all the movies have been chosen
        if i == n :
            return reward, regret
        print(a_t)
        non_select[movieList[a_t]] = False
        x_t_at = movies_feature[a_t]     
        z_t_at =  np.outer(x_t_at, user_feature).flatten()
        A_at_inv = np.linalg.inv(A[a_t])
        #Update global rewards
        A0 = A0 + B[a_t].T.dot(A_at_inv).dot(B[a_t])
        b0 = (b0.T + B[a_t].T.dot(A_at_inv).dot(b[a_t])).T
        #Update local rewards
        rating = user_rating.iloc[[a_t]].rating.values[0]
        reward.append(rating)
        regret.append(user_rating.rating.max()-rating)
        A[a_t] = A[a_t] +  np.outer(x_t_at, x_t_at)
        B[a_t] = B[a_t] +  np.outer(x_t_at, z_t_at)
        b[a_t] = b[a_t] + rating * x_t_at 
        #Update shared features
        A0 = A0 + np.outer(z_t_at,z_t_at) - (B[a_t].T).dot(A_at_inv).dot(B[a_t])
        b0 = (b0.T + rating * z_t_at - (B[a_t].T).dot(A_at_inv).dot(b[a_t])).T
    return reward, regret



#Test Hybrid LinUCB
T = 3000
alpha = 1.5
start_time = time.time()
reward, regret = hybridLinUCB (movies_feature, user_feature, movieList, user_rating, alpha, T)
total_time = time.time() - start_time#3558.683432340622

#Calculate the cumulative regret and reward
cum_reward = []
cum_regret = []
for i in range(len(reward)):
  cum_reward.append(sum(reward[0:i+1]))
for i in range(len(regret)):
  cum_regret.append(sum(regret[0:i+1]))

### Plots
label = 'alpha = ' + str(alpha)
plt.plot(cum_regret, label = label)
plt.xlim(0,T)
plt.xlabel("Step")
plt.ylim(0)
plt.ylabel("Regret")
plt.title("Hybrid LinUCB : Evolution of the regret over time")
plt.show()

#Test Hybrid LinUCB (Prohibit to choose the same movie)
movies_to_keep_1 = pd.DataFrame(ratings.movieId.value_counts()).loc[pd.DataFrame(ratings.movieId.value_counts())['movieId']>=1000].index
df1 = ratings[ratings.movieId.isin(movies_to_keep)]
users_to_keep = pd.DataFrame(df1.userId.value_counts()).loc[pd.DataFrame(df1.userId.value_counts())['userId']>=1000].index
df1 = df1[df1.userId.isin(users_to_keep)]
nbUser = df1.userId.value_counts().shape[0]
nbMovie = df1.movieId.value_counts().shape[0]

user = 89464
index_user = 0
userList = df.userId.unique()
for i in range(len(userList)):
    if userList[i] == user:
        index_user = i
        break

movieList1 = df1[df1['userId']==user].movieId.tolist()
df1 = df1.loc[df1['movieId'].isin(movieList1)]
#keep a record of the ratings given by the user
user_rating1 = df1[df1.userId == user]
#we suppose that this user joined in and had no ratings 
#df.drop(df[df['userId'] == user ].index, inplace = True)
R1 = df1.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
M1 = R1.to_numpy()

from sklearn.decomposition import NMF
model = NMF(n_components=10, init='random', random_state=0, max_iter=5000)
W1 = model.fit_transform(M1) #feauture user
H1 = model.components_ #feature movie
err = np.linalg.norm(M1 - W1 @ H1)**2/np.linalg.norm(M1)**2 
users_feature1 = W1
movies_feature1 = H1.T
user_feature1 = W1[index_user]


T = 1500
alpha = 1.6
start_time = time.time()
reward, regret = hybridLinUCB_bis(movies_feature1, user_feature1, movieList1, user_rating1, alpha, T)
total_time1 = time.time() - start_time#3558.683432340622

#Calculate the cumulative regret and reward
cum_reward = []
cum_regret = []
for i in range(len(reward)):
  cum_reward.append(sum(reward[0:i+1]))
for i in range(len(regret)):
  cum_regret.append(sum(regret[0:i+1]))

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(cum_regret, color='red')
# set x-axis label
ax.set_xlabel("Step")
# set y-axis label
ax.set_ylabel("Regret", color='red')

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
x = np.linspace(0, T, T)
ax2.scatter(x, reward, color='blue')
ax2.set_ylabel("Selected movie rating", color="blue")
plt.title("Hybrid LinUCB : Evolution of the regret over time by preventing user from selecting the same movie")
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg', format='jpeg', dpi=100, bbox_inches='tight')




