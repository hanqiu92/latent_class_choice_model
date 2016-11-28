import numpy as np
import scipy.optimize as opt
import pickle

data = pickle.load( open( "data/quebec_numerical_2.p", "rb" ) )
label = pickle.load( open( "data/quebec_label_2.p", "rb" ) )
state = pickle.load( open( "data/quebec_state.p", "rb" ) )

N, n, m = data.shape
M = state.shape[1]

def get_prob_c(alpha):
    global state
    K = alpha.shape[0]
    util_class = np.sum(alpha.reshape((K,1,M)) * state,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)
    return prob_class

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

def L(w):
    K = len(w) / (m + M)
    alpha = np.reshape(w[0:(M * K)],(K,M))
    beta = np.reshape(w[(M * K):],(K,m))

    prob_c = get_prob_c(alpha)
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_X = np.maximum(np.sum(prob_c * prob_a,axis=0),1e-15)

    out = np.sum(-np.log(prob_X))
    return out

def grad(w):
    global state
    global data
    global label
    K = len(w) / (m + M)
    alpha = np.reshape(w[0:(M * K)],(K,M))
    beta = np.reshape(w[(M * K):],(K,1,1,m))

    prob_c = get_prob_c(alpha)
    prob_i = get_prob_i(beta)
    prob_a = get_prob_a(prob_i)
    prob_X = np.maximum(np.sum(prob_c * prob_a,axis=0),1e-15)
    prob_x = prob_c * prob_a / prob_X

    grad_alpha = np.sum( (prob_c - prob_x).reshape((K,N,1)) * state, axis = 1)
    grad_beta = np.sum( prob_x.reshape((K,N,1)) * np.sum((prob_i - label).reshape((K,N,n,1)) * data, axis = 2 ) , axis = 1)

    out = np.concatenate( (np.reshape(grad_alpha,(M * K)),
                           np.reshape(grad_beta,(m * K))) )
    if np.dot(out,out) > 1e0:
        out = out / np.sqrt(np.dot(out,out))
    return out

Nfeval = 1
def callback_print(w):
    global Nfeval
    if (Nfeval % 100 == 0):
        print Nfeval,':',L(w)
    Nfeval += 1

def post_mnl(K = 1):
    weight = np.zeros(M * K + m * K,np.float64)
    loss = L(weight)
    print 'initial: ',loss
    epsilon = 1.0 / np.sqrt(N) * 1e0
    res = opt.minimize(L,weight,jac = grad,
                       callback = callback_print,
                       options = {'maxiter':5000, 'disp':True})
    weight = res.x
    loss = L(weight)
    print 'final loss:',loss

    alpha = np.reshape(weight[0:(M * K)],(K,M))
    beta = np.reshape(weight[(M * K):],(K,m))
    print 'classifcator:',alpha
    print 'beta',beta

    cla = get_prob_c(alpha)
    print 'classification',np.max(cla,axis=1),np.min(cla,axis=1)

    pickle.dump( {'alpha':alpha,'beta':beta}, open( "result/quebec_final_weight_post_mnl_"+str(K)+".p", "wb" ) )

post_mnl(2)
