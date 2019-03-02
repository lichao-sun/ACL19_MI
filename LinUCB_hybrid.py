import numpy as np


global k
k = 36
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
        x = articles.apply(get_article_features, axis=1).as_matrix()
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

        self.recommIndex = p.argmax()
        chosen = articles[articles.index == articles.index[self.recommIndex]]

        r = self.recommIndex
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
        self.b[self.recommIndex] = self.b[self.recommIndex] + rating * article_features
        return super(hybrid_LinUCB,self).update_features(user_features, article_features, rating, t)

    def compute_utility(self, user_features, article_features, epoch, s):
        return user_features.dot(article_features)