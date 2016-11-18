import numpy as np
import scipy.optimize as opt
import pickle

cate_name = pickle.load( open( "data/quebec_category_name.p", "rb" ) )
cate_data = pickle.load( open( "data/quebec_category.p", "rb" ) )
numer_data = pickle.load( open( "data/quebec_numerical.p", "rb" ) )
label = pickle.load( open( "data/quebec_label.p", "rb" ) )

def likelihood(w, data, label):
    num_labels, sample_size, num_vari = data.shape
    util = np.zeros((num_labels,sample_size),np.float64)
    for i in xrange(num_labels):
        util[i,:] = np.dot(data[i,:,:],w[i*num_vari:(i+1)*num_vari])
    util = np.exp(util - util.max(axis=0))
    prob = util / util.sum(axis=0)
    prob = np.sum(prob * label,axis=0)
    out = np.sum(-np.log(prob))
    return out

def l_gradient(w, data, label):
    num_labels, sample_size, num_vari = data.shape
    out = np.zeros((num_labels,num_vari),np.float64)
    util = np.zeros((num_labels,sample_size),np.float64)
    for i in xrange(num_labels):
        util[i,:] = np.dot(data[i,:,:],w[i*num_vari:(i+1)*num_vari])
    util = np.exp(util - util.max(axis=0))
    prob = util / util.sum(axis=0)
    for i in xrange(num_labels):
        out[i,:] = np.sum(data[i,:,:].T * (prob[i,:] - label[i,:]),axis=1)
    out = np.reshape(out,(num_vari*num_labels))
    if np.dot(out,out) > 1e-8:
        out = out / np.sqrt(np.dot(out,out))
    return out

def p_gradient(w, data, label, epsilon):
    num_labels, sample_size, num_vari = data.shape
    grad_temp = np.zeros((sample_size,num_labels,num_vari),np.float64)
    out = np.ndarray(shape=(sample_size,num_labels*num_vari),dtype=np.float64)
    util = np.zeros((num_labels,sample_size),np.float64)
    for i in range(num_labels):
        util[i,:] = np.dot(data[i,:,:],w[i*num_vari:(i+1)*num_vari])
    util = np.exp(util - util.max(axis=0))
    prob = util / util.sum(axis=0)
    for i in xrange(num_labels):
        grad_temp[:,i,:] = (data[i,:,:].T * (prob[i,:] - label[i,:])).T
    for n in xrange(sample_size):
        grad = np.reshape(grad_temp[n,:,:],(num_vari*num_labels))
        if np.dot(grad,grad) > 1e-8:
            grad = grad / np.sqrt(np.dot(grad,grad))
        out[n,:] = grad
    return out

num_labels, sample_size, num_vari = numer_data.shape
weight_init = np.zeros(num_vari * num_labels,np.float64)
loss = likelihood(weight_init, numer_data, label)
delta_loss = loss
print 'initial: ',loss
delta_bound = delta_loss * 1e-3
epsilon = 1.0 / np.sqrt(sample_size) * 1e0
class_size = 1
class_max = 10
data_list = [numer_data]
weight_list = [weight_init]
label_list = [label]
category_list = [cate_data]
iter_count = 0

while (delta_loss > delta_bound) and (class_size < class_max):
    # first step: calculate MLE for each class
    loss_old = loss
    loss = 0
    print 'iteration ',iter_count,':'
    for i in xrange(class_size):
        res = opt.minimize(likelihood,weight_list[i],jac = l_gradient,
                                       args = (data_list[i],label_list[i]))
        weight_list[i] = res.x
        temp_loss = likelihood(weight_list[i],data_list[i],label_list[i])
        loss += temp_loss
        print 'class ',i,' loss:',temp_loss
    print 'total loss:',loss
    delta_loss = loss_old - loss

    # second step: calculate pointwise gradient (transformed)
    # third step: classification with categorical data
    max_i = -1
    max_rate = 0
    for i in xrange(class_size):
        category_label = p_gradient(weight_list[i],data_list[i],label_list[i],epsilon)
        category_data = category_list[i]
        category_variable_size = category_data.shape[1]
        # cate_data with shape (n,m_c); cate_label with shape (n,m_w)
        max_v = np.sum(np.abs(np.sum(category_label,axis=0)))
        max_c = -1
        max_t = -1
        for c in xrange(category_variable_size):
            type_set = np.unique(category_data[:,c])
            for t in type_set:
                v = 0
                v += np.sum(np.abs(np.sum(category_label[category_data[:,c]<=t,:],axis=0)))
                v += np.sum(np.abs(np.sum(category_label[category_data[:,c]>t,:],axis=0)))
                if v > max_v:
                    max_v = v
                    max_c = c
                    max_t = t

        rate = float(max_v) - float(np.sum(np.abs(np.sum(category_label,axis=0))))
        if rate > max_rate:
            max_rate = rate
            max_i = i

    if max_i >= 0:
        category_data = category_list[max_i]
        ind_1 = category_data[:,max_c]<=max_t
        ind_2 = category_data[:,max_c]>max_t
        if (np.sum(ind_1)>0) and (np.sum(ind_2)>0):
            class_size += 1

            data_temp = data_list[max_i]
            data_list.append(data_temp[:,ind_1,:])
            data_list.append(data_temp[:,ind_2,:])
            del data_list[max_i]
            del data_temp

            weight_list.append(weight_list[max_i])
            weight_list.append(weight_list[max_i])
            del weight_list[max_i]

            label_temp = label_list[max_i]
            label_list.append(label_temp[:,ind_1])
            label_list.append(label_temp[:,ind_2])
            del label_list[max_i]
            del label_temp

            category_temp = category_list[max_i]
            category_list.append(category_temp[ind_1,:])
            category_list.append(category_temp[ind_2,:])
            del category_list[max_i]
            del category_temp

            print 'iteration ',iter_count,': category=',cate_name[max_c],', boundary value=',max_t
        else:
            delta_loss = 0
    else:
        delta_loss = 0

    iter_count += 1
