import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import pickle
import time

data = pickle.load( open( "data/quebec_numerical_2.p", "rb" ) )
label = pickle.load( open( "data/quebec_label_2.p", "rb" ) )
state = pickle.load( open( "data/quebec_state.p", "rb" ) )

N, n, m = data.shape
M = state.shape[1]

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

def get_prob_i(beta):
    global data
    K = len(beta) / m
    util = np.sum(beta.reshape((K,1,1,m)) * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape((K,N,1)))
    prob_individual = util / util.sum(axis=2).reshape((K,N,1))
    return prob_individual

def get_prob_a(prob_i):
    global label
    return np.sum(prob_i * label,axis=2)

def get_prob_X(pi,prob_inv_c):
    return np.sum(pi * prob_inv_c,axis=0)

def get_prob_Y(pi,prob_inv_c,prob_a):
    return np.sum(pi * prob_inv_c * prob_a,axis=0)

def get_prob_y(pi,prob_inv_c,prob_a):
    prob_Y = np.maximum(np.sum(pi * prob_inv_c * prob_a,axis=0), 1e-15)
    prob_y = pi * prob_inv_c * prob_a / prob_Y
    return prob_y

def L(beta, pi, prob_inv_c, log_prob_inv_c):
    K = len(beta) / m
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_X = get_prob_X(pi,prob_inv_c)
    prob_Y = get_prob_Y(pi,prob_inv_c,prob_a)
    out = np.sum(-np.log(prob_Y)+np.log(prob_X))
    return out

def L_f(beta, pi, prob_inv_c, log_prob_inv_c):
    K = len(beta) / m
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_Y = get_prob_Y(pi,prob_inv_c,prob_a)
    out = np.sum(-np.log(prob_Y)-np.max(log_prob_inv_c, axis = 0))
    return out

def grad(beta, pi, prob_inv_c, log_prob_inv_c):
    global data
    global label
    K = len(beta) / m
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_y = get_prob_y(pi,prob_inv_c,prob_a)

    grad_b = - np.sum(prob_y.reshape((K,N,1)) * np.sum((label - prob_i).reshape((K,N,n,1)) * data, axis = 2),axis = 1)
    out = np.reshape(grad_b,(m * K))
    if np.sqrt(np.dot(out,out)) > 1e0:
        out = out / np.sqrt(np.dot(out,out)) * 1e0
    return out

def update(beta, prob_inv_c, pi, theta, Lambda):
    global state
    K = len(beta) / m
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_y = get_prob_y(pi,prob_inv_c,prob_a)

    prob_de = np.maximum(np.sum(prob_y,axis = 1),1e-15)
    pi = prob_de.reshape((K,1)) / float(N)
    theta = np.sum(prob_y.reshape((K,N,1)) * state,axis = 1) / prob_de.reshape((K,1))
    temp = state - theta.reshape((K,1,M))
    Lambda = np.sum(prob_y.reshape((K,N,1,1)) * temp.reshape((K,N,M,1)) * temp.reshape((K,N,1,M)),axis = 1) / prob_de.reshape((K,1,1))
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        if min(v) < 1e-3:
            v = np.maximum(v,1e-3)
            Lambda[k,:,:] = np.matmul(U, np.matmul(np.diag(v),U.T))

    return pi, theta, Lambda

def pre_mnl(K = 1):
    # initiation
    beta = np.zeros(m * K,np.float64)
    pi = 1.0 / float(K) * np.ones((K,1))
    theta = np.zeros((K,M),np.float64)
    Lambda = np.zeros((K,M,M),np.float64)
    choice = np.random.choice(K,N)
    for k in xrange(K):
        ind = np.argwhere(choice == k)
        ind = ind.reshape((len(ind)))
        sample = state[ind,:]
        theta[k,:] = np.mean(sample,axis=0)
        Lambda[k,:,:] = np.mean((sample-theta[k,:]).reshape((len(ind),M,1)) * (sample-theta[k,:]).reshape((len(ind),1,M)),axis=0)
        v,U = la.eigh(Lambda[k,:,:])
        if min(v) < 1e-3:
            v = np.maximum(v,1e-3)
            Lambda[k,:,:] = np.matmul(U, np.matmul(np.diag(v),U.T))
    prob_inv_c, log_prob_inv_c = get_prob_inv_c(theta,Lambda)
    pi, theta, Lambda = update(beta, prob_inv_c, pi, theta, Lambda)

    # begin optimization
    t = time.clock()
    n_iter = 0
    while (n_iter < 200):
        prob_inv_c, log_prob_inv_c = get_prob_inv_c(theta,Lambda)
        if (n_iter % 10 == 0):
            print n_iter,', L:',L(beta, pi, prob_inv_c, log_prob_inv_c),', L_f:',L_f(beta, pi, prob_inv_c, log_prob_inv_c)
        #'''
        res = opt.minimize(L_f,beta,jac = grad,
                           args = (pi, prob_inv_c, log_prob_inv_c),
                           options = {'maxiter':10, 'disp':False})
        beta = res.x
        #'''
        '''
        for i in xrange(10):
            grad_beta = grad(beta, pi, prob_inv_c, log_prob_inv_c)
            beta -= 100.0/(n_iter+i/10.0+100.0) * grad_beta
        '''
        pi, theta, Lambda = update(beta, prob_inv_c, pi, theta, Lambda)
        n_iter += 1

    prob_inv_c, log_prob_inv_c = get_prob_inv_c(theta,Lambda)
    print 'final L:',L(beta, pi, prob_inv_c, log_prob_inv_c),', L_f:',L_f(beta, pi, prob_inv_c, log_prob_inv_c)
    print 'total time:',time.clock() - t

    prob_X = np.maximum(np.sum(pi * prob_inv_c,axis=0),1e-15)
    prob_x = pi * prob_inv_c / prob_X
    print 'classification',np.max(prob_x,axis=1),np.min(prob_x,axis=1)

    print 'pi:',pi
    print 'GMM:',theta
    print 'beta:',beta.reshape((K,m))

    pickle.dump( {'beta':beta.reshape((K,m)),'pi':pi,'theta':theta,'Lambda':Lambda}, open( "result/quebec_final_weight_pre_mnl_"+str(K)+".p", "wb" ) )

pre_mnl(2)
