import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

A=np.array([[ 0.70558142,  0.86273594],
       [ 0.68797076,  0.02774431],
       [ 0.11372903,  0.38687036],
       [ 0.43072205,  0.54141448],
       [ 0.39529217,  0.91495635],
       [ 0.39027663,  0.95704016],
       [ 0.5450535,  0.02473527],
       [ 0.12265648,  0.88966732],
       [ 0.59852203,  0.28636077],
       [ 0.85040799,  0.04557076],
       [ 0.56896747,  0.89559782],
       [ 0.85425338,  0.01838715],
       [ 0.80782139,  0.17350079]])
#plt.boxplot(A)

a_mu=np.mean(A,axis=0)
a_sig=np.std(A,axis=0)
a_upper_ci=a_mu + (3 * a_sig)
print a_upper_ci

n=5000  # number of data points
k=30   # number of features
n_a=8  # number of actions

D=np.random.random( (n,k))-0.5            # our data
th=np.random.random( (n_a,k) ) - 0.5      # our real theta, what we will try to guess/


P=D.dot(th.T)
print P[0]
import matplotlib.pyplot as plt
optimal=np.array(P.argmax(axis=1), dtype=int)
plt.title("Distribution of ideal arm choices")
plt.hist(optimal,bins=range(0,n_a))

eps=0.2

choices=np.zeros(n)
rewards=np.zeros(n)
explore=np.zeros(n)
norms  =np.zeros(n)
b      =np.zeros_like(th)
A      =np.zeros( (n_a, k,k)  )
for a in range (0,n_a):
    A[a]=np.identity(k)
th_hat =np.zeros_like(th) # our temporary feature vectors, our best current guesses
p      =np.zeros(n_a)
alph   =0.2


# LinUCB, using a disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit approach..." Li, Langford
for i in range(0,n):

    x_i = D[i]   # the current context vector

    for a in range (0,n_a):
        A_inv      = np.linalg.inv(A[a])        # we use it twice so cache it
        th_hat[a]  = A_inv.dot(b[a])            # Line 5
        ta         = x_i.dot(A_inv).dot(x_i)    # how informative is this ?
        a_upper_ci = alph * np.sqrt(ta)         # upper part of variance interval
        a_mean     = th_hat[a].dot(x_i)         # current estimate of mean
        p[a]       = a_mean + a_upper_ci        # top CI


    norms[i]       = np.linalg.norm(th_hat - th,'fro')    # diagnostic, are we converging ?

    # Let's not be biased with tiebreaks, but add in some random noise
    p= p + ( np.random.random(len(p)) * 0.000001)
    choices[i] = p.argmax()   # choose the highest, line 11
    a = int(choices[i])
    # see what kind of result we get
    rewards[i] = th[a].dot(x_i)  # using actual theta to figure out reward

    # update the input vector
    A[a]      += np.outer(x_i,x_i)
    b[a]      += rewards[i] * x_i

plt.figure(1,figsize=(10,5))
plt.subplot(121)
plt.plot(norms);
plt.title("Frobenius norm of estimated theta vs actual")
plt.show()

regret=(P.max(axis=1) - rewards)
plt.subplot(122)
plt.plot(regret.cumsum())
plt.title("Cumulative Regret")
plt.show()