import numpy as np
import numpy.random as rd
import numpy.linalg as la
import scipy.optimize as opt
import pickle
import time
nrand = rd.multivariate_normal

data = pickle.load( open( "data/quebec_numerical_2.p", "rb" ) )
label = pickle.load( open( "data/quebec_label_2.p", "rb" ) )
state = pickle.load( open( "data/quebec_state.p", "rb" ) )

N, n, m = data.shape
M = state.shape[1]
S = 100

def get_beta(mu,Sigma):
    K = mu.shape[0]
    beta = np.zeros(shape=(S,K,N,m),dtype=np.float64)
    for k in xrange(K):
        beta[:,k,:,:] = nrand(mu[k,:],Sigma[k,:,:],(S,N))
    return beta

def get_prob_c(alpha):
    global state
    K = len(alpha) / M
    util_class = np.sum(alpha.reshape((K,1,M)) * state,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    return util_class / util_class.sum(axis=0)

def get_prob_s(beta):
    global data
    global label
    K = beta.shape[1]
    util = np.sum(beta.reshape(S,K,N,1,m) * data, axis=4)
    util = np.exp(util - util.max(axis=3).reshape(S,K,N,1))
    prob_individual = util / util.sum(axis=3).reshape(S,K,N,1)
    return np.sum(prob_individual * label,axis=3)

def get_prob_X(prob_c,prob_s):
    return np.sum(prob_c * prob_s,axis=(0,1)) / float(S)

def get_prob_x(prob_c,prob_s):
    prob_X = np.sum(prob_c * prob_s,axis=(0,1))
    prob_x = prob_c * prob_s / prob_X.reshape((1,1,N))
    return prob_x

def L(alpha, prob_s):
    prob_c = get_prob_c(alpha)
    prob_X = get_prob_X(prob_c,prob_s)
    out = np.sum(-np.log(prob_X))
    return out

def grad(alpha, prob_s):
    global state
    K = len(alpha) / M
    prob_c = get_prob_c(alpha)
    prob_x = get_prob_x(prob_c,prob_s)

    grad_a = - np.sum((np.sum(prob_x,axis=0) - prob_c).reshape(K,N,1) * state,axis=1)
    grad_a = grad_a.reshape((K * M))
    if np.sum(grad_a * grad_a) > 1e0:
        grad_a = grad_a / np.sqrt(np.sum(grad_a * grad_a))
    return grad_a

def update(alpha, prob_s, beta, mu, Sigma):
    K = len(alpha) / M
    prob_c = get_prob_c(alpha)
    prob_x = get_prob_x(prob_c,prob_s)

    prob_de = np.maximum(np.sum(prob_x,axis = (0,2)),1e-15)
    mu = np.sum(prob_x.reshape(S,K,N,1) * beta,axis = (0,2)) / prob_de.reshape(K,1)
    temp = beta - mu.reshape(K,1,m)

    Sigma = np.zeros((K,m,m),np.float64)
    for s in xrange(S):
        Sigma += np.sum( prob_x[s,:,:].reshape(K,N,1,1) * temp[s,:,:,:].reshape(K,N,m,1) * temp[s,:,:,:].reshape(K,N,1,m),axis = 1 )
    Sigma = Sigma / prob_de.reshape((K,1,1))
    for k in xrange(K):
        v,U = la.eigh(Sigma[k,:,:])
        if min(v) < 0:
            v = np.maximum(v,0)
            Sigma[k,:,:] = np.matmul(U, np.matmul(diag(v), U.T))

    return mu,Sigma

def post_mixture(K = 1):
    #alpha = np.zeros(M * K,np.float64)
    #mu = np.zeros((K,m),np.float64)

    weight = pickle.load( open( "result/quebec_final_weight_post_mnl_"+str(K)+".p", "rb" ) )
    alpha = weight[0:(M * K)]
    mu = weight[(M * K):((M + m) * K)].reshape((K,m))
    Sigma = np.zeros((K,m,m),np.float64)
    for k in xrange(K):
        Sigma[k,:,:] = 1.0 * np.eye(m)

    t = time.clock()
    n_iter = 0
    while (n_iter < 200):
        beta = get_beta(mu,Sigma)
        prob_s = get_prob_s(beta)
        if (n_iter % 10 == 0):
            print n_iter,',pre L:',L(alpha,prob_s)

        #'''
        res = opt.minimize(L,alpha,jac = grad,
                           args = (prob_s),
                           options = {'maxiter':10, 'disp':False})
        alpha = res.x
        #'''
        '''
        for i in xrange(10):
            grad_a = grad(alpha,prob_s)
            alpha -= 100.0 / (n_iter + i/10.0 + 100.0) * grad_a
        '''

        mu,Sigma = update(alpha,prob_s,beta,mu,Sigma)
        if (n_iter % 10 == 0):
            print n_iter,',post L:',L(alpha,prob_s)
        n_iter += 1

    beta = get_beta(mu,Sigma)
    prob_s = get_prob_s(beta)
    loss = L(alpha,prob_s)
    print 'final loss:',loss
    print 'total time:',time.clock() - t

    print 'classifcator:',alpha
    cla = get_prob_c(alpha)
    print 'classification',np.max(cla,axis=1),np.min(cla,axis=1)

    print 'mu:',mu
    print 'Sigma:',Sigma

    pickle.dump( np.concatenate((alpha,mu.reshape((K * m)),Sigma.reshape((K * m * m)))), open( "result/quebec_final_weight_post_mixture_"+str(K)+".p", "wb" ) )

post_mixture(2)
