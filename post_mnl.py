import numpy as np
import scipy.optimize as opt
import pickle

data = pickle.load( open( "data/quebec_numerical_2.p", "rb" ) )
label = pickle.load( open( "data/quebec_label_2.p", "rb" ) )
state = pickle.load( open( "data/quebec_state.p", "rb" ) )

N, n, m = data.shape
M = state.shape[1]

def L(w, data, state, label):
    K = len(w) / (m + M)
    a = np.reshape(w[0:(M * K)],(K,1,M))
    b = np.reshape(w[(M * K):],(K,1,1,m))

    util_class = np.sum(a * state,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape(K,N,1))
    prob = util / util.sum(axis=2).reshape(K,N,1)
    prob = np.sum(prob * label,axis=2)
    prob = np.sum(prob_class * prob,axis=0)
    out = np.sum(-np.log(prob))
    return out

def grad(w, data, state, label):
    K = len(w) / (m + M)
    a = np.reshape(w[0:(M * K)],(K,1,M))
    b = np.reshape(w[(M * K):],(K,1,1,m))

    util_class = np.sum(a * state,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=2).reshape(K,N,1))
    prob_individual = util / util.sum(axis=2).reshape(K,N,1)
    prob_assort = np.sum(prob_individual * label,axis=2)
    prob = np.sum(prob_class * prob_assort,axis=0)

    grad_a = np.zeros((K,1,M),np.float64)
    grad_b = np.zeros((K,1,1,m),np.float64)

    for k in xrange(K):
        grad_a[k,0,:] = np.sum( (prob_class[k,:] * (1.0 - prob_assort[k,:] / prob)).reshape((N,1)) * state, axis = 0)
        grad_b[k,0,0,:] = np.sum( (1.0 / prob * prob_class[k,:] * prob_assort[k,:]).reshape((N,1)) * np.sum((prob_individual[k,:,:] - label).reshape((N,n,1)) * data, axis = 1 ) , axis = 0)

    out = np.concatenate( (np.reshape(grad_a,(M * K)),
                           np.reshape(grad_b,(m * K))) )

    if np.dot(out,out) > 1e-8:
        out = out / np.sqrt(np.dot(out,out))
    return out

def get_class(w, data, state, label):
    K = len(w) / (m + M)
    a = np.reshape(w[0:(M * K)],(K,1,M))
    b = np.reshape(w[(M * K):],(K,1,1,m))

    util_class = np.sum(a * state,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)

    return prob_class

Nfeval = 1
def callback_print(w):
    global Nfeval
    if (Nfeval % 100 == 0):
        print Nfeval,':',L(w,data,state,label)
    Nfeval += 1

def post_mnl(K = 1):
    weight = np.zeros(M * K + m * K,np.float64)
    loss = L(weight, data, state, label)
    print 'initial: ',loss
    epsilon = 1.0 / np.sqrt(N) * 1e0
    res = opt.minimize(L,weight,jac = grad,
                       args = (data,state,label),
                       callback = callback_print,
                       options = {'maxiter':5000, 'disp':True})
    weight = res.x
    loss = L(weight,data,state,label)
    print 'final loss:',loss

    print 'classifcator:',np.reshape(weight[0:M * K],(K,M))

    cla = get_class(weight,data,state,label)
    print 'classification',np.max(cla,axis=1),np.min(cla,axis=1)

    print 'beta',weight[(M * K):]
    #pickle.dump( weight, open( "result/quebec_final_weight_post_mnl_"+str(K)+".p", "wb" ) )

post_mnl(2)
