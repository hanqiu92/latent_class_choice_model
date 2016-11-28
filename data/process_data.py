import numpy as np
import pickle

f_raw = open('quebec.dat','rb')
sample_size = 3090
cate_list = [0,3,4,5,8]
cate_name = ['sector','conv_year','house_type','constr_year','own_rent']
cate_data = np.ndarray(shape=(sample_size,len(cate_list)),dtype=np.int64)
shared_list = [1,6,7,9,10,11]
num_shared = 6
shared_data = np.ndarray(shape=(sample_size,len(shared_list)),dtype=np.float64)
num_labels = 9
num_individual = 4
individual_data = np.ndarray(shape=(num_labels,sample_size,num_individual),dtype=np.float64)
label_i = np.ndarray(shape=(sample_size),dtype=np.int64)
label = np.zeros(shape=(num_labels,sample_size),dtype=np.float64)
count = 0

for line in f_raw:
    if count > 0:
        data = line.strip().split("\t")
        # sector, hdd, choice, conv_year, house_type, constr_year, nb_rooms, nb_pers,
        #    own_rent, surface, age, income = data[:12]
        # op_cost = data[12:21]
        # fix_cost = data[21:30]
        # cost_inc = data[30:39]
        # avail = data[39:48]
        for i in xrange(len(cate_list)):
            cate_data[count-1,i] = int(data[cate_list[i]])
        for i in xrange(len(shared_list)):
            shared_data[count-1,i] = float(data[shared_list[i]])
        for i in xrange(num_individual):
            for j in xrange(num_labels):
                individual_data[j,count-1,i] = float(data[12+i*num_labels+j])
        label_i[count-1] = int(data[2])-1
        label[int(data[2])-1,count-1] = 1.0
    count += 1

f_raw.close()

num_vari_1 = len(shared_list)+num_individual+1
numer_data_1 = np.ones(shape=(num_labels,sample_size,num_vari_1),dtype=np.float64)
numer_data_1[:,:,(len(shared_list)+1):] = individual_data
for i in xrange(num_labels):
    numer_data_1[i,:,1:(len(shared_list)+1)] = shared_data

num_vari_2 = num_labels * num_shared+num_individual+1
numer_data_2 = np.zeros(shape=(sample_size,num_labels,num_vari_2),dtype=np.float64)
for i in xrange(num_labels):
    numer_data_2[:,i,0] = np.zeros((sample_size),dtype=np.float64)
    numer_data_2[:,i,(num_labels*num_shared+1):] = individual_data[i,:,:]
    numer_data_2[:,i,(i*num_shared+1):((i+1)*num_shared+1)] = shared_data

num_state = num_shared+num_labels * num_individual+1
state = np.zeros(shape=(sample_size,num_state),dtype=np.float64)
state[:,0:num_shared] = shared_data
for i in xrange(num_labels):
    state[:,(num_shared+i*num_individual):(num_shared+(i+1)*num_individual)] = individual_data[i,:,:]

pickle.dump( cate_name, open( "quebec_category_name.p", "wb" ) )
pickle.dump( cate_data, open( "quebec_category.p", "wb" ) )
pickle.dump( shared_data, open( "quebec_shared.p", "wb" ) )
pickle.dump( individual_data, open( "quebec_individual.p", "wb" ) )
pickle.dump( numer_data_1, open( "quebec_numerical_1.p", "wb" ) )
pickle.dump( numer_data_2, open( "quebec_numerical_2.p", "wb" ) )
pickle.dump( state, open( "quebec_state.p", "wb" ) )
pickle.dump( label, open( "quebec_label_1.p", "wb" ) )
pickle.dump( label.T, open( "quebec_label_2.p", "wb" ) )
