import numpy as np
import scipy.optimize as opt
import pickle

data = pickle.load( open( "data/quebec_numerical_2.p", "rb" ) )
label = pickle.load( open( "data/quebec_label_2.p", "rb" ) )
state = pickle.load( open( "data/quebec_state.p", "rb" ) )

N, n, m = data.shape
M = state.shape[1] + 1
state = np.hstack((np.ones((N,1)),state))

def get_prob_inv_c(alpha):
    global state
    K = alpha.shape[0]
    log_prob_class = - np.square(np.sum(alpha.reshape((K,1,M)) * state,axis = 2))
    prob_class = np.exp(log_prob_class - log_prob_class.max(axis=0))
    return prob_class, log_prob_class

def get_prob_i(beta):
    global data
    K = beta.shape[0]
    util = np.sum(beta.reshape((K,1,1,m)) * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape((K,N,1)))
    prob_individual = util / util.sum(axis=2).reshape((K,N,1))
    return prob_individual

def get_prob_a(prob_i):
    global label
    return np.sum(prob_i * label,axis=2)

def get_prob_X(prob_c,prob_inv_c):
    return np.sum(prob_c * prob_inv_c,axis=0)

def get_prob_Y(prob_c,prob_inv_c,prob_a):
    return np.sum(prob_c * prob_inv_c * prob_a,axis=0)

def get_prob_y(prob_c,prob_inv_c,prob_a):
    prob_Y = np.maximum(np.sum(prob_c * prob_inv_c * prob_a,axis=0), 1e-15)
    prob_y = prob_c * prob_inv_c * prob_a / prob_Y
    return prob_y

def L(w):
    K = len(w) / (1 + m + M)
    pi = np.reshape(w[0:K],(K,1))
    alpha = np.reshape(w[K:(K + M * K)],(K,M))
    beta = np.reshape(w[(K + M * K):],(K,m))

    pi = np.exp(pi)
    prob_c = pi / np.sum(pi)
    prob_inv_c,log_prob_inv_c = get_prob_inv_c(alpha)
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_X = get_prob_X(prob_c,prob_inv_c)
    prob_Y = get_prob_Y(prob_c,prob_inv_c,prob_a)
    out = np.sum(-np.log(prob_Y)+np.log(prob_X))
    return out

def L_f(w):
    K = len(w) / (1 + m + M)
    pi = np.reshape(w[0:K],(K,1))
    alpha = np.reshape(w[K:(K + M * K)],(K,M))
    beta = np.reshape(w[(K + M * K):],(K,m))

    pi = np.exp(pi)
    prob_c = pi / np.sum(pi)
    prob_inv_c,log_prob_inv_c = get_prob_inv_c(alpha)
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_Y = get_prob_Y(prob_c,prob_inv_c,prob_a)
    out = np.sum(-np.log(prob_Y)-np.max(log_prob_inv_c, axis = 0))
    return out

def grad(w):
    global state
    global data
    global label
    K = len(w) / (1 + m + M)
    pi = np.reshape(w[0:K],(K,1))
    alpha = np.reshape(w[K:(K + M * K)],(K,M))
    beta = np.reshape(w[(K + M * K):],(K,m))

    pi = np.exp(pi)
    prob_c = pi / np.sum(pi)
    prob_inv_c,log_prob_inv_c = get_prob_inv_c(alpha)
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_y = get_prob_y(prob_c,prob_inv_c,prob_a)

    grad_pi = prob_c - np.sum( prob_y, axis = 1 ).reshape((K,1)) / N
    grad_alpha = 2 * np.sum( prob_y.reshape((K,N,1)) * np.sum(alpha.reshape((K,1,M)) * state,axis = 2).reshape((K,N,1)) * state, axis = 1)
    grad_beta = np.sum( prob_y.reshape((K,N,1)) * np.sum((prob_i - label).reshape((K,N,n,1)) * data, axis = 2 ) , axis = 1)

    out = np.concatenate( (np.reshape(grad_pi,(K)),
                           np.reshape(grad_alpha,(M * K)),
                           np.reshape(grad_beta,(m * K))) )
    if np.dot(out,out) > 1e0:
        out = out / np.sqrt(np.dot(out,out))
    return out

Nfeval = 1
def callback_print(w):
    global Nfeval
    if (Nfeval % 100 == 0):
        print Nfeval,': L,',L(w),',L_f,',L_f(w)
    Nfeval += 1

def pre_2_mnl(K = 1):
    weight = np.random.rand(K + M * K + m * K) - 0.5
    print 'initial L: ',L(weight),', L_f: ',L_f(weight)
    epsilon = 1.0 / np.sqrt(N) * 1e0
    res = opt.minimize(L_f,weight,jac = grad,
                       callback = callback_print,
                       options = {'maxiter':5000, 'disp':True})
    weight = res.x
    print 'final L: ',L(weight),', L_f: ',L_f(weight)

    pi = np.reshape(weight[0:K],(K,1))
    pi_temp = np.exp(pi)
    prob_c = pi_temp / np.sum(pi_temp)
    alpha = np.reshape(weight[K:(K + M * K)],(K,M))
    beta = np.reshape(weight[(K + M * K):],(K,m))
    print 'class:',prob_c
    print 'classifcator:',alpha
    print 'beta',beta

    prob_inv_c,log_prob_inv_c = get_prob_inv_c(alpha)
    prob_X = np.maximum(np.sum(prob_c * prob_inv_c,axis=0),1e-15)
    cla = prob_c * prob_inv_c / prob_X
    print 'classification',np.max(cla,axis=1),np.min(cla,axis=1)

    pickle.dump( {'pi':pi,'alpha':alpha,'beta':beta}, open( "result/quebec_final_weight_pre_2_mnl_"+str(K)+".p", "wb" ) )

pre_2_mnl(10)
