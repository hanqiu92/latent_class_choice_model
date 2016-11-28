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

def get_prob_inv_c(theta,Lambda):
    global state
    K = theta.shape[0]
    log_prob_inv_class = np.zeros((K,N),np.float64)
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        d = state - theta[k,:]
        d = np.matmul(np.matmul(U, np.matmul(np.diag(1.0/np.sqrt(v)),U.T)), d.T)
        log_prob_inv_class[k,:] = - 0.5 * (sum(np.log(v)) + np.sum(d * d,axis = 0))
    prob_inv_class = np.exp(log_prob_inv_class - np.max(log_prob_inv_class, axis = 0))
    return prob_inv_class, log_prob_inv_class

def get_prob_s(beta):
    global data
    global label
    K = beta.shape[1]
    util = np.sum(beta.reshape((S,K,N,1,m)) * data, axis=4)
    util = np.exp(util - util.max(axis=3).reshape((S,K,N,1)))
    prob_individual = util / util.sum(axis=3).reshape((S,K,N,1))
    return np.sum(prob_individual * label,axis=3)

def get_prob_X(pi,prob_inv_c):
    return np.sum(pi * prob_inv_c,axis=0)

def get_prob_Y(pi,prob_inv_c,prob_s):
    return np.sum(pi * prob_inv_c * prob_s,axis=(0,1)) / float(S)

def get_prob_y(pi,prob_inv_c,prob_s):
    prob_Y = np.maximum(np.sum(pi * prob_inv_c * prob_s,axis=(0,1)), 1e-15)
    prob_y = pi * prob_inv_c * prob_s / prob_Y
    return prob_y

def L(pi, prob_inv_c, prob_s):
    prob_X = get_prob_X(pi,prob_inv_c)
    prob_Y = get_prob_Y(pi,prob_inv_c,prob_s)
    out = np.sum(-np.log(prob_Y)+np.log(prob_X))
    return out

def L_f(pi, prob_inv_c, log_prob_inv_c, prob_s):
    prob_Y = get_prob_Y(pi,prob_inv_c,prob_s)
    out = np.sum(-np.log(prob_Y)-np.max(log_prob_inv_c, axis = 0))
    return out

def update(prob_inv_c, prob_s, beta, pi, theta, Lambda, mu, Sigma):
    K = pi.shape[0]
    prob_y = get_prob_y(pi,prob_inv_c,prob_s)

    prob_de = np.maximum(np.sum(prob_y,axis = (0,2)),1e-15)
    pi = prob_de.reshape((K,1)) / float(N)
    theta = np.sum(np.sum(prob_y,axis=0).reshape((K,N,1)) * state,axis = 1) / prob_de.reshape((K,1))
    mu = np.sum(prob_y.reshape((S,K,N,1)) * beta,axis = (0,2)) / prob_de.reshape((K,1))
    temp1 = state - theta.reshape((K,1,M))
    Lambda = np.sum(np.sum(prob_y,axis=0).reshape((K,N,1,1)) * temp1.reshape((K,N,M,1)) * temp1.reshape((K,N,1,M)), axis = 1) / prob_de.reshape((K,1,1))

    temp2 = beta - mu.reshape((K,1,m))
    Sigma = np.zeros((K,m,m),np.float64)
    for s in xrange(S):
        Sigma += np.sum( prob_y[s,:,:].reshape((K,N,1,1)) * temp2[s,:,:,:].reshape((K,N,m,1)) * temp2[s,:,:,:].reshape((K,N,1,m)),axis = 1 )
    Sigma = Sigma / prob_de.reshape((K,1,1))

    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        if min(v) < 1e-3:
            v = np.maximum(v,1e-3)
            Lambda[k,:,:] = np.matmul(U, np.matmul(np.diag(v), U.T))

        v,U = la.eigh(Sigma[k,:,:])
        if min(v) < 0:
            v = np.maximum(v,0)
            Sigma[k,:,:] = np.matmul(U, np.matmul(np.diag(v), U.T))

    return pi, theta, Lambda, mu, Sigma

def pre_mixture(K = 1):
    '''
    pi = np.ones((K,1),np.float64) / float(K)
    theta = np.zeros((K,M),np.float64)
    mu = np.zeros((K,m),np.float64)
    Lambda = np.zeros((K,M,M),np.float64)
    Sigma = np.zeros((K,m,m),np.float64)
    for k in xrange(K):
        Lambda[k,:,:] = 1.0 * np.eye(M)
        Sigma[k,:,:] = 1.0 * np.eye(m)
    '''
    #'''
    weight = pickle.load( open( "result/quebec_final_weight_pre_mnl_"+str(K)+".p", "rb" ) )
    mu = weight['beta']
    pi = weight['pi']
    theta = weight['theta']
    Lambda = weight['Lambda']
    Sigma = np.zeros((K,m,m),np.float64)
    for k in xrange(K):
        Sigma[k,:,:] = 1.0 * np.eye(m)
    #'''

    t = time.clock()
    n_iter = 0
    while (n_iter < 200):
        prob_inv_c, log_prob_inv_c = get_prob_inv_c(theta,Lambda)
        beta = get_beta(mu,Sigma)
        prob_s = get_prob_s(beta)
        if (n_iter % 1 == 0):
            print n_iter,', L:',L(pi, prob_inv_c, prob_s),', L_f:',L_f(pi, prob_inv_c, log_prob_inv_c, prob_s)

        pi, theta, Lambda, mu, Sigma = update(prob_inv_c, prob_s, beta, pi, theta, Lambda, mu, Sigma)
        n_iter += 1

    prob_inv_c,log_prob_inv_c = get_prob_inv_c(theta,Lambda)
    beta = get_beta(mu,Sigma)
    prob_s = get_prob_s(beta)
    print 'final L:',L(pi, prob_inv_c, prob_s),', L_f:',L_f(pi, prob_inv_c,log_prob_inv_c, prob_s)
    print 'total time:',time.clock() - t

    prob_X = np.maximum(np.sum(pi * prob_inv_c,axis=0),1e-15)
    prob_x = pi * prob_inv_c / prob_X
    print 'classification',np.max(prob_x,axis=1),np.min(prob_x,axis=1)

    print 'pi:',pi
    print 'GMM:',theta
    print 'mu:',mu
    print 'Sigma:',Sigma

    pickle.dump( {'pi':pi,'theta':theta,'Lambda':Lambda,'mu':mu,'Sigma':Sigma}, open( "result/quebec_final_weight_pre_mixture_"+str(K)+".p", "wb" ) )

pre_mixture(2)
