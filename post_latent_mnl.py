import numpy as np
import scipy.optimize as opt
import pickle

data = pickle.load( open( "data/quebec_numerical.p", "rb" ) )
label = pickle.load( open( "data/quebec_label.p", "rb" ) )
shared_data = pickle.load( open( "data/quebec_shared.p", "rb" ) )

def likelihood(w, data, shared_data, label):
    num_labels, sample_size, num_vari = data.shape
    num_shared = shared_data.shape[1]
    num_class = len(w) / (num_vari * num_labels + num_shared)

    a = np.reshape(w[0:num_shared * num_class],(num_class,1,num_shared))
    b = np.reshape(w[num_shared * num_class:],(num_class,num_labels,1,num_vari))

    util_class = np.sum(a * shared_data,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=1).reshape(num_class,1,sample_size))
    prob = util / util.sum(axis=1).reshape(num_class,1,sample_size)
    prob = np.sum(prob * label,axis=1)
    prob = np.sum(prob_class * prob,axis=0)
    out = np.sum(-np.log(prob))
    return out

def l_gradient(w, data, shared_data, label):
    num_labels, sample_size, num_vari = data.shape
    num_shared = shared_data.shape[1]
    num_class = len(w) / (num_vari * num_labels + num_shared)

    a = np.reshape(w[0:num_shared * num_class],(num_class,1,num_shared))
    b = np.reshape(w[num_shared * num_class:],(num_class,num_labels,1,num_vari))

    util_class = np.sum(a * shared_data,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)

    util = np.sum(b * data, axis=3)
    util = np.exp(util - util.max(axis=1).reshape(num_class,1,sample_size))
    prob_individual = util / util.sum(axis=1).reshape(num_class,1,sample_size)
    prob_assort = np.sum(prob_individual * label,axis=1)
    prob = np.sum(prob_class * prob_assort,axis=0)

    grad_a = np.zeros((num_class,1,num_shared),np.float64)
    grad_b = np.zeros((num_class,num_labels,1,num_vari),np.float64)

    for k in xrange(num_class):
        grad_a[k,0,:] = np.sum( (prob_class[k,:] * (1.0 - prob_assort[k,:] / prob)).reshape((sample_size,1)) * shared_data, axis = 0)
        for i in xrange(num_labels):
            grad_b[k,i,0,:] = np.sum( 1.0 / prob * prob_class[k,:] * prob_individual[k,i,:] * (prob_assort[k,:] - label[i,:]) * data[i,:,:].T, axis = 1 )

    out = np.concatenate( (np.reshape(grad_a,(num_shared * num_class)),
                           np.reshape(grad_b,(num_vari * num_labels * num_class))) )

    if np.dot(out,out) > 1e-8:
        out = out / np.sqrt(np.dot(out,out))
    return out

def get_class(w, data, shared_data, label):
    num_labels, sample_size, num_vari = data.shape
    num_shared = shared_data.shape[1]
    num_class = len(w) / (num_vari * num_labels + num_shared)

    a = np.reshape(w[0:num_shared * num_class],(num_class,1,num_shared))
    b = np.reshape(w[num_shared * num_class:],(num_class,num_labels,1,num_vari))

    util_class = np.sum(a * shared_data,axis = 2)
    util_class = np.exp(util_class - util_class.max(axis=0))
    prob_class = util_class / util_class.sum(axis=0)

    return prob_class

def sim_likelihood(w, data, label):
    num_labels, sample_size, num_vari = data.shape
    b = np.reshape(w,(num_labels,1,num_vari))
    util = np.sum(b * data, axis=2)
    util = np.exp(util - util.max(axis=0))
    prob = util / util.sum(axis=0)
    prob = np.sum(prob * label,axis=0)
    out = np.sum(-np.log(prob))
    return out

def sim_l_gradient(w, data, label):
    num_labels, sample_size, num_vari = data.shape
    b = np.reshape(w,(num_labels,1,num_vari))
    util = np.sum(b * data, axis=2)
    util = np.exp(util - util.max(axis=0))
    prob_individual = util / util.sum(axis=0)
    prob_assort = np.sum(prob_individual * label,axis=0)

    grad_b = np.zeros((num_labels,1,num_vari),np.float64)
    for i in xrange(num_labels):
        grad_b[i,0,:] = np.sum(data[i,:,:].T * prob_individual[i,:] * (1.0 - label[i,:] / prob_assort),axis=1)

    out = np.reshape(grad_b,(num_vari*num_labels))
    if np.dot(out,out) > 1e-8:
        out = out / np.sqrt(np.dot(out,out))
    return out


Nfeval = 1
def callback_print(w):
    global Nfeval
    print Nfeval,':',likelihood(w,data,shared_data,label)
    Nfeval += 1

def post_latent_mnl(num_class = 1):
    num_labels, sample_size, num_vari = data.shape
    num_shared = shared_data.shape[1]
    weight = np.zeros(num_shared * num_class + num_vari * num_labels * num_class,np.float64)
    loss = likelihood(weight, data, shared_data, label)
    print 'initial: ',loss
    epsilon = 1.0 / np.sqrt(sample_size) * 1e0
    res = opt.minimize(likelihood,weight,jac = l_gradient,
                       args = (data,shared_data,label),
                       #callback = callback_print,
                       options = {'maxiter':2000, 'disp':True})
    weight = res.x
    loss = likelihood(weight,data,shared_data,label)
    print 'final loss:',loss

    print 'classifcator:',np.reshape(weight[0:num_shared * num_class],(num_class,num_shared))

    cla = get_class(weight,data,shared_data,label)
    print 'classification',cla
    '''
    cla = cla > (1.0 / float(num_class))
    print 'total',np.sum(cla,axis=1)
    pickle.dump(cla,open('data/classification.p','wb'))

    for i in xrange(num_class):
        c = cla[i,:]
        print 'cal class ',i+1,' loss:',sim_likelihood(weight[(num_shared * num_class + num_vari * num_labels * i):(num_shared * num_class + num_vari * num_labels * (i+1))],data[:,c,:],label[:,c])
        weight_s = np.zeros(num_vari * num_labels,np.float64)
        res = opt.minimize(sim_likelihood,weight_s,jac = sim_l_gradient,
                           args = (data[:,c,:],label[:,c]))
        print 're cal class ',i+1,' loss:',sim_likelihood(res.x,data[:,c,:],label[:,c])
    '''

post_latent_mnl(2)
