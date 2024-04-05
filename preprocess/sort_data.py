import numpy as np
import os

# this file takes .csv files and make it to matrices sorted geometrically and saves them as .npz files


#makes a list of columns to use from
def get_xyz_shape(experiment):
    num_gridpoints = experiment.shape[0]
    for i in range(num_gridpoints-1):
        if experiment[i,0] != experiment[i+1,0]:
            yz_plane = i+1
            break

    return yz_plane


def check_sorting(experiment,dim):
    num_gridpoints = experiment.shape[0]
    for i in range(num_gridpoints-1):
        if experiment[i,dim] > experiment[i+1,dim]:
            print(f'Error in row {i}')

        if i % 10000 == 0:
            print(f'Iterationstep {i}')
    print('done')


# creates three lists with all coordinate values for x,y,z
def xyz_coordinates(exp):
    x,y,z = [],[],[]
    num_gridpoints= exp.shape[0]
    for i in range(num_gridpoints):
        if exp[i,0] not in x:
            x.append(exp[i,0])
        if exp[i,1] not in y:
            y.append(exp[i,1])
        if exp[i,2] not in z:
            z.append(exp[i,2])
    n_points = len(x)*len(y)*len(z)

    return x,y,z,n_points

def replace_coordinates(exp):
    for i in range(exp.shape[0]):
        exp[i,[0,1,2]] = exp[i,[0,1,2]]*10
    return exp


def create_inputmatrix(exp,x,y,z,time):
    inputmatrix = np.empty(shape=(x,y,z))
    for i in range(exp.shape[0]): #iterates through all gridpoints
        inputmatrix[int(exp[i,0]),int(exp[i,1]),int(exp[i,2])] = exp[i,time+4]

    return inputmatrix



n_time = 60 #number of timesteps

for dirname in os.listdir('data/new'):
    if dirname != 'Mesh_files':
        exp_1 = np.loadtxt('./data/new/'+dirname+'/experiment_df.csv',
                                delimiter=',', dtype = np.float32, skiprows=1, usecols = np.arange(1,n_time+5)) # loads dataset from .csv as an array of float32 entries
                                                                                                        # and uses cols 1 to 64
        # loads an array of all (# gridpoints, 64) where 64 are x,y,z coordinate, iblank, temp in 60 timesteps
        n = exp_1.shape[0]
        print ('Number rows in the experiment_df file: {}'.format(n))
        print (f'Number of columns in exp = {exp_1.shape[1]}')

        #multiplies coordinates with 10 such that you get integers for the matrix
        exp_new_coord = replace_coordinates(exp_1)

        #get lists of x,y,z values and the actual number of gridpoints (earlier there was double countings from different room quarters)
        x,y,z,num_gridpoints = xyz_coordinates(exp_new_coord)

        print(f'x={len(x)}\ny={len(y)}\nz={len(z)}')

        #translate the .csv rows into matrix entries, for each time_step one matrix
        matrix_temp_list =[]
        for time in range(n_time):
            matrix_temp_list.append([])
            print(f'time = {time}')
            matrix_temp_list[time] = create_inputmatrix(exp_new_coord, len(x),len(y),len(z),time)
            print('matrix t_{} was calculated'.format(time))


        #np.savez('./data/new/'+dirname+'/temperature_matrices.npz',matrix_t0 =matrix_temp_list[0],matrix_t1= matrix_temp_list[1],matrix_t2=matrix_temp_list[2],
        #         matrix_t3=matrix_temp_list[3],matrix_t4=matrix_temp_list[4],matrix_t5=matrix_temp_list[5],matrix_t6=matrix_temp_list[6],
        #         matrix_t7=matrix_temp_list[7],matrix_t8=matrix_temp_list[8], matrix_t9=matrix_temp_list[9], matrix_t10=matrix_temp_list[10],
        #         matrix_t11=matrix_temp_list[11])

        np.savez('./data/new/'+dirname+'/temperature_matrices.npz', matrix_t0 = matrix_temp_list[0], matrix_t1 = matrix_temp_list[1], matrix_t2 = matrix_temp_list[2], matrix_t3 = \
        matrix_temp_list[3], matrix_t4 = matrix_temp_list[4], matrix_t5 = matrix_temp_list[5], matrix_t6 = \
        matrix_temp_list[6], matrix_t7 = matrix_temp_list[7], matrix_t8 = matrix_temp_list[8], matrix_t9 = \
        matrix_temp_list[9], matrix_t10 = matrix_temp_list[10], matrix_t11 = matrix_temp_list[11], matrix_t12 = \
        matrix_temp_list[12], matrix_t13 = matrix_temp_list[13], matrix_t14 = matrix_temp_list[14], matrix_t15 = \
        matrix_temp_list[15], matrix_t16 = matrix_temp_list[16], matrix_t17 = matrix_temp_list[17], matrix_t18 = \
        matrix_temp_list[18], matrix_t19 = matrix_temp_list[19], matrix_t20 = matrix_temp_list[20], matrix_t21 = \
        matrix_temp_list[21], matrix_t22 = matrix_temp_list[22], matrix_t23 = matrix_temp_list[23], matrix_t24 = \
        matrix_temp_list[24], matrix_t25 = matrix_temp_list[25], matrix_t26 = matrix_temp_list[26], matrix_t27 = \
        matrix_temp_list[27], matrix_t28 = matrix_temp_list[28], matrix_t29 = matrix_temp_list[29], matrix_t30 = \
        matrix_temp_list[30], matrix_t31 = matrix_temp_list[31], matrix_t32 = matrix_temp_list[32], matrix_t33 = \
        matrix_temp_list[33], matrix_t34 = matrix_temp_list[34], matrix_t35 = matrix_temp_list[35], matrix_t36 = \
        matrix_temp_list[36], matrix_t37 = matrix_temp_list[37], matrix_t38 = matrix_temp_list[38], matrix_t39 = \
        matrix_temp_list[39], matrix_t40 = matrix_temp_list[40], matrix_t41 = matrix_temp_list[41], matrix_t42 = \
        matrix_temp_list[42], matrix_t43 = matrix_temp_list[43], matrix_t44 = matrix_temp_list[44], matrix_t45 = \
        matrix_temp_list[45], matrix_t46 = matrix_temp_list[46], matrix_t47 = matrix_temp_list[47], matrix_t48 = \
        matrix_temp_list[48], matrix_t49 = matrix_temp_list[49], matrix_t50 = matrix_temp_list[50], matrix_t51 = \
        matrix_temp_list[51], matrix_t52 = matrix_temp_list[52], matrix_t53 = matrix_temp_list[53], matrix_t54 = \
        matrix_temp_list[54], matrix_t55 = matrix_temp_list[55], matrix_t56 = matrix_temp_list[56], matrix_t57 = \
        matrix_temp_list[57], matrix_t58 = matrix_temp_list[58], matrix_t59 = matrix_temp_list[59])

        # smoother version would be:
        # np.savez_compressed('./data/new/'+dirname+'/temperature_matrices.npz', **data_arrays)
        print(matrix_temp_list[0].shape)










#check_sorting(exp_1,0)


#miniexp = exp_1[:3,:]
#
#sorted_x_exp = np.sort(exp_1, axis=0)

#check_sorting(sorted_exp,0)

#print(get_xyz_shape(sorted_x_exp))



