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

def L(beta, data, state, label, parameters):
    pi = parameters['pi']
    theta = parameters['theta']
    Lambda = parameters['Lambda']
    K = len(beta) / m
    b = np.reshape(beta,(K,1,1,m))

    log_prob_inv_class = np.zeros((K,N),np.float64)
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        d = state - theta[k,:]
        d = np.matmul(np.matmul(U, np.matmul(np.diag(1.0/np.sqrt(v)),U.T)), d.T)
        log_prob_inv_class[k,:] = - 0.5 * (sum(np.log(v)) + np.sum(d * d,axis = 0))
    prob_inv_class = np.exp(log_prob_inv_class - np.max(log_prob_inv_class, axis = 0))

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape(K,N,1))
    prob_individual = util / util.sum(axis=2).reshape(K,N,1)
    prob_assort = np.sum(prob_individual * label,axis=2)

    prob_X = np.maximum(np.sum(pi * prob_inv_class,axis=0),1e-15)
    prob_Y = np.maximum(np.sum(pi * prob_inv_class * prob_assort,axis=0),1e-15)

    out = np.sum(-np.log(prob_Y / prob_X))
    return out

def L_f(beta, data, state, label, parameters):
    pi = parameters['pi']
    theta = parameters['theta']
    Lambda = parameters['Lambda']
    K = len(beta) / m
    b = np.reshape(beta,(K,1,1,m))

    log_prob_inv_class = np.zeros((K,N),np.float64)
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        d = state - theta[k,:]
        d = np.matmul(np.matmul(U, np.matmul(np.diag(1.0/np.sqrt(v)),U.T)), d.T)
        log_prob_inv_class[k,:] = - 0.5 * (sum(np.log(v)) + np.sum(d * d,axis = 0))
    prob_inv_class = np.exp(log_prob_inv_class - np.max(log_prob_inv_class, axis = 0))

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape(K,N,1))
    prob_individual = util / util.sum(axis=2).reshape(K,N,1)
    prob_assort = np.sum(prob_individual * label,axis=2)

    prob_Y = np.maximum(np.sum(pi * prob_inv_class * prob_assort,axis=0),1e-15)
    out = np.sum(-np.log(prob_Y) - np.max(log_prob_inv_class, axis = 0))
    return out

def grad(beta, data, state, label, parameters):
    pi = parameters['pi']
    theta = parameters['theta']
    Lambda = parameters['Lambda']
    K = len(beta) / m
    b = np.reshape(beta,(K,1,1,m))

    log_prob_inv_class = np.zeros((K,N),np.float64)
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        d = state - theta[k,:]
        d = np.matmul(np.matmul(U, np.matmul(np.diag(1.0/np.sqrt(v)),U.T)), d.T)
        log_prob_inv_class[k,:] = - 0.5 * (sum(np.log(v)) + np.sum(d * d,axis = 0))
    prob_inv_class = np.exp(log_prob_inv_class - np.max(log_prob_inv_class,axis = 0))

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape(K,N,1))
    prob_individual = util / util.sum(axis=2).reshape(K,N,1)
    prob_assort = np.sum(prob_individual * label,axis=2)

    prob_Y = np.maximum(np.sum(pi * prob_inv_class * prob_assort,axis=0),1e-15)
    prob_y = (pi * prob_inv_class * prob_assort) / prob_Y

    grad_b = - np.sum(prob_y.reshape((K,N,1)) * np.sum((label - prob_individual).reshape(K,N,n,1) * data, axis = 2),axis = 1)

    out = np.reshape(grad_b,(m * K))
    if np.sqrt(np.dot(out,out)) > 1e0:
        out = out / np.sqrt(np.dot(out,out)) * 1e0
    return out

def update(beta, data, state, label, parameters):
    pi = parameters['pi']
    theta = parameters['theta']
    Lambda = parameters['Lambda']
    K = len(beta) / m
    b = np.reshape(beta,(K,1,1,m))

    log_prob_inv_class = np.zeros((K,N),np.float64)
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        d = state - theta[k,:]
        d = np.matmul(np.matmul(U, np.matmul(np.diag(1.0/np.sqrt(v)),U.T)), d.T)
        log_prob_inv_class[k,:] = - 0.5 * (sum(np.log(v)) + np.sum(d * d,axis = 0))
    prob_inv_class = np.exp(log_prob_inv_class - np.max(log_prob_inv_class, axis = 0))

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape(K,N,1))
    prob_individual = util / util.sum(axis=2).reshape(K,N,1)
    prob_assort = np.sum(prob_individual * label,axis=2)

    prob_Y = np.maximum(np.sum(pi * prob_inv_class * prob_assort,axis=0),1e-15)
    prob_y = (pi * prob_inv_class * prob_assort) / prob_Y

    pi = (np.sum(prob_y,axis=1)).reshape((K,1)) / float(N)
    if min(pi) > 1e-15:
        theta = np.sum(prob_y.reshape(K,N,1) * state,axis = 1) / np.sum(prob_y.reshape(K,N,1),axis = 1)
        for k in xrange(K):
            g = state - theta[k,:]
            Lambda[k,:,:] = np.sum(prob_y[k,:].reshape(N,1,1) * (g.reshape(N,M,1) * g.reshape(N,1,M)),axis = 0) / np.sum(prob_y[k,:])
            v,U = la.eigh(Lambda[k,:,:])
            if min(v) < -1e-3:
                print 'singular covariance!'
                print 'v = ',min(v)
                v = np.maximum(v,1e-3)
                Lambda[k,:,:] = np.matmul(U, np.matmul(np.diag(v),U.T))
            elif min(v) < 1e-3:
                v = np.maximum(v,1e-3)
                Lambda[k,:,:] = np.matmul(U, np.matmul(np.diag(v),U.T))

    parameters = {'pi':pi,'theta':theta,'Lambda':Lambda}

def get_class(beta, data, state, label, parameters):
    pi = parameters['pi']
    theta = parameters['theta']
    Lambda = parameters['Lambda']
    K = len(beta) / m
    b = np.reshape(beta,(K,1,1,m))

    log_prob_inv_class = np.zeros((K,N),np.float64)
    for k in xrange(K):
        v,U = la.eigh(Lambda[k,:,:])
        d = state - theta[k,:]
        d = np.matmul(np.matmul(U, np.matmul(np.diag(1.0/np.sqrt(v)),U.T)), d.T)
        log_prob_inv_class[k,:] = 0.5 * (sum(np.log(v)) + np.sum(d * d,axis = 0))
    prob_inv_class = np.exp(log_prob_inv_class - np.max(log_prob_inv_class, axis = 0))

    prob_X = np.maximum(np.sum(pi * prob_inv_class,axis=0),1e-15)
    prob_x = (pi * prob_inv_class) / prob_X

    return prob_x

def pre_mnl(K = 1):
    beta = np.zeros(m * K,np.float64)
    #w_init = pickle.load( open( "result/quebec_final_weight_post_mnl_"+str(K)+".p", "rb" ) )
    #beta = w_init[(M * K):]

    # initiation
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

    parameters = {'pi':pi,'theta':theta,'Lambda':Lambda}
    update(beta,data,state,label,parameters)

    print 'initial L: ',L(beta,data,state,label,parameters),', L_f:',L_f(beta,data,state,label,parameters)
    t = time.clock()

    n_iter = 0
    while (n_iter < 200):
        #'''
        res = opt.minimize(L_f,beta,jac = grad,
                           args = (data,state,label,parameters),
                           options = {'maxiter':10, 'disp':False})
        beta = res.x
        #'''
        '''
        for i in xrange(10):
            grad_beta = grad(beta,data,state,label,parameters)
            beta -= 100.0/(n_iter+i/10.0+100.0) * grad_beta
        '''
        update(beta,data,state,label,parameters)
        n_iter += 1
        if (n_iter % 1 == 0):
            print n_iter,', L:',L(beta,data,state,label,parameters),', L_f:',L_f(beta,data,state,label,parameters)

    print 'final L:',L(beta,data,state,label,parameters),', L_f:',L_f(beta,data,state,label,parameters)
    print 'total time:',time.clock() - t

    print 'GMM:',np.reshape(theta,(K,M))

    cla = get_class(beta,data,state,label,parameters)
    print 'classification',np.max(cla,axis=1),np.min(cla,axis=1)

    pickle.dump( np.concatenate((beta,parameters['pi'].reshape((K)),parameters['theta'].reshape(M * K),parameters['Lambda'].reshape(M * M * K))), open( "result/quebec_final_weight_pre_mnl_"+str(K)+".p", "wb" ) )

pre_mnl(1)
