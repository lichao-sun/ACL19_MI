import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
import os
import random
import math
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import pymc3 as pm
import theano
import datetime
theano.config.compute_test_value = 'raise'
#matplotlib inline
from multiprocessing import Pool, cpu_count
from functools import partial


SELECTED_DATA_DIR = "../selected-data/"
MOVIES_FILE = "best_movie_ratings_features_engineered.csv"
USERS_FILE = "users_ratings.csv"

movies_raw = pd.read_csv(SELECTED_DATA_DIR + MOVIES_FILE, index_col=0)
movies_raw.rating = movies_raw.rating/10
movies_raw.sample()

users = pd.read_csv(SELECTED_DATA_DIR + USERS_FILE, index_col=0)
users.rating = users.rating/10
users.sample()

WANTED_DIM = 20

pca_df = movies_raw[list(movies_raw.columns[2:])]
pca = PCA(n_components=WANTED_DIM)
pca_df = pd.DataFrame(pca.fit_transform(pca_df))
pca_df.index = movies_raw.index

movies = pd.concat([movies_raw[list(movies_raw.columns[:2])], pd.DataFrame(pca_df)] ,axis=1)

collabo = movies.merge(users, left_index=True, right_index=True)

for n in range(WANTED_DIM):
    collabo[n] = (collabo[n] * collabo['rating_x'])* collabo['rating_x'] # fois le rating au carre
    print collabo['rating_x']

collabo = collabo.groupby(collabo.user).aggregate(np.average)

for n in range(WANTED_DIM):
    collabo[n] = (collabo[n] * collabo['rating_x']) # fois le rating moyen pour pouvoir compare les users

collabo = collabo[[n for n in range(WANTED_DIM)]]


articles = movies
print articles.sample(3)
print users.sample(3)
# collabo = articles.merge(users, left_index=True, right_index=True)
# print collabo.sample(5)
# print collabo.shape
#
# print collabo['rating']
#
# for n in range(WANTED_DIM):
#     collabo[n] = (collabo[n] * collabo['rating'])* collabo['rating']
#
# collabo = collabo.groupby(collabo.user_id).aggregate(np.average)
#
# for n in range(WANTED_DIM):
#     collabo[n] = (collabo[n] * collabo['rating'])
#
# collabo = collabo[[n for n in range(WANTED_DIM)]]
#
# print collabo.sample()

__metaclass__ = type
class algorithm:
    def update_features(self, user_features, movie_features, rating, t):
        return update_features(user_features, movie_features, rating, t)
    def compute_utility(self, user_features, movie_features, epoch, s):
        return compute_utility(user_features, movie_features, epoch, s)

class random_choice(algorithm):
    def choice(self, user_features, movies, epoch, s):
        """ random approach to the problem, always exploring"""
        return movies.sample()

def greedy_choice_t(user_features, movies, epoch, s, recommf):
    """ greedy with decreasing epsilon """
    epsilon = 1 / math.sqrt(epoch+1)
    return greedy_choice_no_t(user_features, movies, epoch, s, recommf, epsilon)

def greedy_choice_no_t(user_features, movies, epoch, s, recommf, epsilon):
    """ greedy with fixed epsilon """
    if random.random() > epsilon:
        return recommf(user_features, movies, epoch, s)
    else:
        return movies.sample()

class greedy_choice_contentbased(algorithm):
    def choice(self, user_features, movies, epoch, s):
        """ greedy approach to the problem """
        return greedy_choice_t(user_features, movies, epoch, s, best_contentbased_recommandation)

class greedy_choice_no_t_contentbased(algorithm):
    def choice(self, user_features, movies, epoch, s, epsilon=0.3):
        """ greedy approach to the problem """
        return greedy_choice_no_t(user_features, movies, epoch, s, best_contentbased_recommandation, epsilon)

class greedy_choice_UCB(algorithm):
    def choice(self, user_features, movies, epoch, s):
        """ greedy approach with upper confidence bounds """
        return greedy_choice_t(user_features, movies, epoch, s, partial(best_contentbased_recommandation, UCB=True))

class greedy_choice_collaborative(algorithm):
    def choice(self, user_features, movies, epoch, s):
        """ greedy approach to the problem """
        return greedy_choice_t(user_features, movies, epoch, s, best_collaborative_recommandation)

class greedy_choice_no_t_collaborative(algorithm):
    def choice(self, user_features, movies, epoch, s, epsilon=0.3):
        """ greedy approach to the problem """
        return greedy_choice_no_t(user_features, movies, epoch, s, best_collaborative_recommandation, epsilon)

class LinUCB(algorithm):
    def __init__(self, alpha):
        self.first = True
        self.alpha = alpha

    def choice(self, user_features, movies, epoch, s):
        # movies features
        x = movies.apply(get_movie_features, axis=1).as_matrix()
        # number of movies
        m = x.shape[0]
        # dimension of movie features
        d = x.shape[1]
        # initialize when first time
        if self.first:
            self.first = False
            self.A = np.zeros((m, d, d))
            for a in range(m):
                self.A[a] = np.eye(d)
            self.b = np.zeros((m, d))
        # get rating for every movie
        ratings = np.zeros(m)

        for a, (title, movie) in enumerate(movies.iterrows()):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])
            ratings[a] = theta_a.T.dot(x[a]) + self.alpha * np.sqrt(x[a].T.dot(A_inv).dot(x[a]))
        self.recomm = ratings.argmax()
        chosen = movies[movies.index == movies.index[self.recomm]]
        self.A[self.recomm] += x[self.recomm].dot(x[self.recomm].T)
        return chosen

    def update_features(self, user_features, movie_features, rating, t):
        self.b[self.recomm] += rating * movie_features
        return super(LinUCB,self).update_features(user_features, movie_features, rating, t)

    def compute_utility(self, user_features, movie_features, epoch, s):
        return user_features.dot(movie_features)

global k
k = 400
class hybrid_LinUCB(algorithm):
    def __init__(self, alpha):
        self.first = True
        self.alpha = alpha
        self.A0 = np.eye(k)
        self.b0 = np.zeros(k)

    def choice(self, user_features, articles, epoch, s):
        """
        user_features: feature vector for the user for whom the next recommendation is being asked
        articles: pool of all articles
        """
        # articles features
        x = articles.apply(get_movie_features, axis=1).as_matrix()
        # number of articles
        m = x.shape[0]
        # dimension of article features
        d = x.shape[1]
        # initialize when first time
        A0_inv = np.linalg.inv(self.A0)
        self.beta = A0_inv.dot(self.b0)
        if self.first:
            self.first = False
            self.A = np.zeros((m, d, d))
            for a in range(m):
                self.A[a] = np.eye(d)
            self.b = np.zeros((m, d))
            self.B = np.zeros((m, d, k))
        # get rating for every article
        p = np.zeros(m)
        s = np.zeros(m)

        Z = {}
        # iterate over each article a, compute its interaction with the user and recommend the highest
        for a, (title, article) in enumerate(articles.iterrows()):
            z_a = np.outer(x[a],user_features).flatten() # z_a contains the passed in method user and the article a's interaction
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a]-self.B[a].dot(self.beta))
            s[a] = z_a.T.dot(A0_inv).dot(z_a) - 2*z_a.T.dot(A0_inv).dot(self.B[a].T).dot(A_inv).dot(x[a]) \
                + x[a].T.dot(A_inv).dot(x[a]) + x[a].T.dot(A_inv).dot(self.B[a]).dot(A0_inv).dot(self.B[a].T).dot(A_inv).dot(x[a])
            p[a] = z_a.T.dot(self.beta) + x[a].T.dot(theta_a) + self.alpha * np.sqrt(s[a])
            #print z_a
            Z[a] = z_a

        self.recomm = p.argmax()
        chosen = articles[articles.index == articles.index[self.recomm]]

        r = self.recomm
        A_inv_recomm = np.linalg.inv(self.A[r])
        self.A0 = self.A0 + self.B[r].T.dot(A_inv_recomm).dot(self.B[r])
        self.b0 = self.b0 + self.B[r].T.dot(A_inv_recomm).dot(self.b[r])
        self.A[r] = self.A[r] + x[r].dot(x[r].T)
        self.B[r] = self.B[r] + np.reshape(x[r],(d,1)).dot(np.reshape(Z[r],(k,1)).T)
        #self.b[r] += p[r].dot(x[r])
        self.A0 = self.A0 + Z[r].dot(Z[r].T) - self.B[r].T.dot(A_inv_recomm).dot(self.B[r])
        self.b0 = self.b0 + p[r]*Z[r] - self.B[r].T.dot(A_inv_recomm).dot(self.b[r])

        return chosen

    def update_features(self, user_features, article_features, rating, t):
        #print rating, article_features
        #print type(rating * article_features)
        #print type(self.b[self.recomm])
        self.b[self.recomm] = self.b[self.recomm] + rating * article_features
        return super(hybrid_LinUCB,self).update_features(user_features, article_features, rating, t)

    def compute_utility(self, user_features, article_features, epoch, s):
        return user_features.dot(article_features)

def bayes_UCB(user_features, movies, epoch, s):
    # Hyperparameters
    c0 = 10
    d0 = 3
    e0 = 0.01
    f0 = 0.001
    g0 = 0.001
    # function
    I = np.eye(user_features.size)
    ratings = np.zeros(movies.shape[0])
    with pm.Model():
        s = pm.Gamma('s', d0, e0)
        sigma = pm.InverseGamma('sigma', f0, g0)
        theta = pm.MvNormal('theta', mu=0.5, cov=c0 * sigma * I)
        rating = pm.Normal('rating', mu=0, sd=sigma, observed=user_features)

        for i, (title, movie) in tqdm(enumerate(movies.iterrows())):
            movies_features = get_movie_features(movies)
            # Expected value of outcome
            mu = user_features.dot(movies_features) * (1 - np.exp(-epoch/s))
            # Likelihood (sampling distribution) of observations
            rating.mu = mu

            step = pm.Metropolis()
            trace = pm.sample(1000, step=step, njobs=1, progressbar=False)
            ratings[i] = rating.distribution.random()[0]
    return movies[movies.index == movies.index[ratings.argmax()]]

def compute_utility(user_features, movie_features, epoch, s):
    """ Compute utility U based on user preferences and movie preferences """
    res = user_features.dot(movie_features) * (1 - math.exp(-epoch/s))
    return res

def compute_novelty(allepoch, s):
    """ Compute utility U based on user preferences and movie preferences """
    res = []
    for epoch in allepoch:
        res.append(1 - math.exp(-epoch/s))
    return res

def compute_UCB(epoch, Nt):
    return math.sqrt((2 * math.log2(epoch + 1)) / (Nt * epoch))

def get_movie_features(movie):
    """ selected features from dataframe """
    if isinstance(movie, pd.Series):
        return movie[-WANTED_DIM:]
    elif isinstance(movie, pd.DataFrame):
        return get_movie_features(movie.loc[movie.index[0]])
    else:
        raise TypeError("{} should be a Series or DataFrame".format(movie))

def iterative_mean(old, new, t):
    """ Compute the new mean """
    return ((t-1) / t) * old + (1/t) * new

def update_features(user_features, movie_features, rating, t):
    """ update the user preferences """
    return iterative_mean(user_features, movie_features * rating, t+1)

def best_contentbased_recommandation(user_features, movies, epoch, s, UCB=False):
    """ Return the movie with the highest utility """
    utilities = np.zeros(movies.shape[0])
    for i, (title, movie) in enumerate(movies.iterrows()):
        movie_features = get_movie_features(movie)
        utilities[i] = compute_utility(user_features, movie_features, epoch - movie.last_t, s)
        if UCB:
            utilities[i] += compute_UCB(epoch, movie.Nt)
    return movies[movies.index == movies.index[utilities.argmax()]]

def best_collaborative_recommandation____(user_features, user_movies, epoch, s):
    """ Return the movie with the highest utility """
    corr = np.zeros(collabo.shape[0])
    corruser = np.zeros(collabo.shape[0])
    # TODO retirer lui-meme de la matrix collabo
    # on fait une pearson corr avec tous les autres users -> CLUSTERING
    for collabi, collabrow in enumerate(collabo.iterrows()):
        otheruser_index = collabrow[0]
        otheruser_features = collabrow[1]
        corr[collabi] = np.correlate(user_features, otheruser_features)
        corruser[collabi] = otheruser_index
    # on prends les films des 5 plus proche
    idxbestuser = []
    for i in range(10):
        idxmax = corr.argmax()
        idxbestuser.append(corruser[idxmax])
        corruser[idxmax] = 0
    moviesbestuser = users.copy()[users.user.isin(idxbestuser)].index
    # on fait une jointure avec les films du user
    try:
        subsetmovie = user_movies.copy().loc[moviesbestuser]
        subsetmovie = subsetmovie.dropna()
    except:
        print("WARNING : no jointure btw user")
        return best_contentbased_recommandation(user_features, user_movies, epoch, s)

    ## TODO : verifier qu'on ne l'a pas deja vu
    #argmaxmovie = subsetmovie['rating'].argmax()
    #if subsetmovie.loc[argmaxmovie][0] == 'rating':
    #    print('WTF')
    #    print(subsetmovie.loc[argmaxmovie].name)
    ##print(subsetmovie.loc[argmaxmovie])
    #return subsetmovie.loc[argmaxmovie]
    return best_contentbased_recommandation(user_features, subsetmovie, epoch, s)

def best_collaborative_recommandation(user_features, user_movies, epoch, s):
    """ Return the movie with the highest utility """
    corr = np.zeros(collabo.shape[0])
    corruser = np.zeros(collabo.shape[0])
    # on fait une pearson corr avec tous les autres users -> CLUSTERING
    for collabi, collabrow in enumerate(collabo.iterrows()):
        otheruser_index = collabrow[0]
        otheruser_features = collabrow[1]
        corr[collabi] = float(np.correlate(user_features, otheruser_features)[0])
        corruser[collabi] = otheruser_index
    # on prends les films des 5 plus proche
    idxbestuser = []
    for i in range(10):
        idxmax = corr.argmax()
        idxbestuser.append(corruser[idxmax])
        corruser[idxmax] = 0
    moviesbestuser = users.copy()[users.user.isin(idxbestuser)].index
    # on fait une jointure avec les films du user
    try:
        subsetmovie = user_movies.copy().loc[moviesbestuser]
        subsetmovie = subsetmovie.dropna()
    except:
        print("WARNING : no jointure btw user")
        return best_contentbased_recommandation(user_features, user_movies, epoch, s)
    subsetmovie['rating'] = subsetmovie['rating'] * compute_novelty(epoch - subsetmovie.last_t, s)
    maxrating = subsetmovie['rating'].max()
    return subsetmovie[subsetmovie.rating == maxrating].sample()

def reinforcement_learning(user, moviestc, algorithm, s, numberSimulation):
    if s<200:
        print("WARNING : s is really small, movies will get often repeated")
    algorithm = algorithm()
    user_features = np.zeros(moviestc.shape[1] - 2)
    movies = moviestc.copy()
    movies = movies[movies.columns.difference(["votes", "rating"])]
    movies.insert(0, 'last_t', np.ones(movies.shape[0]).astype(np.int64))
    movies.insert(0, 't', [i for i in range(movies.shape[0])])
    movies.insert(0, 'rating', user.rating)
    movies.insert(0, 'Nt', np.zeros(movies.shape[0]))
    cumregret = [0]
    accuracy_rmse = [0]
    avg_rating = [0]
    timestamp = []
    for t in range(numberSimulation):
        now = datetime.datetime.now()
        recommandation = algorithm.choice(user_features, movies, t+1, s)
        recommandation_features = get_movie_features(recommandation)
        user_rating = user.loc[recommandation.index[0]].rating
        user_features = algorithm.update_features(user_features, recommandation_features, user_rating, t)
        utility = algorithm.compute_utility(user_features, recommandation_features, t, s)
        cumregret.append(cumregret[-1] + (user_rating - utility ))
        accuracy_rmse.append((user_rating - utility )**2 )
        avg_rating.append(user_rating)
        movies.loc[movies.index.isin(recommandation.index),'last_t'] = t
        movies.loc[movies.index.isin(recommandation.index),'Nt'] += 1
        timestamp.append((datetime.datetime.now() - now).total_seconds())
    return {'cumregret': cumregret, 'accuracy_rmse':accuracy_rmse, 'avg_rating':avg_rating, 'timediff':timestamp}

def rl_multiple_users(users, movies, algorithms, s=500, N=20, N_USER=50):
    def wrapper_rl_one_user(args):
        return reinforcement_learning(*args)
    results_all = []
    users_sample = users[users.user.isin(pd.Series(users.user.unique()).sample(N_USER))].copy()
    movies_sample = movies[movies.index.isin(users_sample.index.unique())].copy()
    for algo in tqdm(algorithms):
        res_algo = []
        args = []
        for i, name in enumerate(users_sample.user.unique()):
            user = users[users.user == name]
            movies_user = movies_sample[movies_sample.index.isin(user.index)]
            res = reinforcement_learning(user, movies_user, algo, s, N)
            res_algo.append(res)
        results_all.append(res_algo)
    return results_all

# Keep list consistent
ALGOS      = [partial(hybrid_LinUCB,0.5), partial(LinUCB, 0.5), greedy_choice_no_t_contentbased, random_choice] #, greedy_choice, random_choice]
ALGOS_NAME = ['hybrid_LinUCB','LinUCB', 'greedy_choice_no_t_contentbased', 'random_choice'] #, 'greedy_choice', 'random_choice']
assert(len(ALGOS) == len(ALGOS_NAME))

METRICS = ['cumregret', 'accuracy_rmse', 'avg_rating', 'timediff']
TITLE_GRAPH=['Average cumulative regret for each algorithm', 'Average accuracy for each algorithm', 'Average accuracy for each algorithm', 'Average running time for each algorithm']
X_AXIS = ['Cumulative reget', 'Accuracy (root mean square error)', 'Rating', 'Time']
assert(len(METRICS) == len(TITLE_GRAPH) == len(X_AXIS))

res = rl_multiple_users(users, movies, ALGOS, s=200, N=500, N_USER=5)

for metric, tgraph, xaxix in zip(METRICS,TITLE_GRAPH,X_AXIS):
    data = []
    for algo, algon in enumerate(ALGOS_NAME):
        temp = np.average(np.array([i[metric] for i in res[algo]]), axis=0)[1:]
        data.append(go.Scatter(
            x = list([i for i in range(len(temp))]),
            y = temp,
            name=algon
        ))
    layout = dict(title = tgraph,
              xaxis = dict(title = tgraph),
              yaxis = dict(title = xaxix),
    )
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig)