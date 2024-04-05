import numpy as np
import os

# takes .npz files and normalises the data

#takes the list of all temperature matrices, calculates global min and max and normalizes each value
def create_min_dist(matrixlist):
    min_abs = 1000000
    max_abs = 0

    for i in range(matrixlist.shape[0]):
        x_min, x_max = np.amin(matrixlist[i]), np.amax(matrixlist[i])
        #print(min, max)
        if x_min<min_abs:
            #print(f'Neues Minimum: {min}')
            min_abs = x_min

        if x_max>max_abs:
            #print(f'Neues Maximum: {max}')
            max_abs=x_max

    dist = max_abs - min_abs

    return min_abs, dist

# normalize the matrixarray with min, dist
def normalize(min_abs,dist, matrixlist):
    for i in range(matrixlist.shape[0]):
        matrixlist[i]= (matrixlist[i]-min_abs)/dist
    return matrixlist


# creates an array matrixlist = [temperaturematrix_t0, ..., temperaturematrix_t59] for one experiment
def npz_to_matrixlist(file):
    npz = np.load(file)
    #print(type(npz))
    #print(np.array(npz).shape[0])
    matrixlist = []
    for i in range(np.array(npz).shape[0]):
        matrixlist.append([])
        matrixlist[i] = np.array([npz['matrix_t'+str(i)]])
    return np.array(matrixlist)


def ableiten(matrixlist):
    abl_list = []
    for i in range(matrixlist.shape[0]-1):
        abl_list.append(matrixlist[i+1]-matrixlist[i])
    return np.array(abl_list)


# this is only needed if i try to derive the dataset
def derive(file):
    xy= npz_to_matrixlist(file)
    xy=ableiten(xy)
    #xy=normalize(xy) # the dataset is normalized over one room
    return xy



# creates a dictionary xy = {experimentname : [temperaturematrix_t0, ... , temperaturematrixt_59]}
# abl = true if derivation is needed, false if the datasest is taken as it is
def import_all_rooms(abl):
    xy = dict()
    #for dirname in os.listdir(r'C:\Users\Artur\Documents\GitHub\geospatial-time-series\data\new'):
    for dirname in os.listdir(r'./data/new/'):
        if dirname != 'Mesh_files':
            #file = "C:\\Users\\Artur\\Documents\\GitHub\\geospatial-time-series\\data\\new\\" + dirname + "\\temperature_matrices.npz"

            file = r'./data/new/' + dirname + r'/temperature_matrices.npz'

            if abl == True:
                xy[dirname] = derive(file)

            else:
                xy[dirname] = npz_to_matrixlist(file)


    ml, timesteps = matrix_list_of_all_experiments(xy)

    min, dist = create_min_dist(ml) #creates absolut minimum of the set and the distance for normalizing later

    print(f"min = {min} \n dist = {dist}")

    # creates the normalized dictionary by using the list of all temp_matrices
    xy_norm = {}
    for i, key in enumerate(xy.keys()):
        array_60_seconds = normalize(min, dist, ml[i * timesteps:(i + 1) * timesteps])
        new_array = np.expand_dims(array_60_seconds, axis=1)
        xy_norm[key] = new_array
    return xy_norm

# creates an array out of all tempmatrices for all timesteps and all experiments: (number of experiments * timesteps, 61,81,31)
# and the number of timesteps
def matrix_list_of_all_experiments(dictionary_xy):
    matrixlist_all = []

    # get number of timesteps in each experiment
    first_key = next(iter(dictionary_xy))   # get the first key of the dict
    first_value = dictionary_xy[first_key]        # get the first value of the dict
    timesteps = len(first_value)

    for key in dictionary_xy:
        matrixlist_all.extend(dictionary_xy[key])
    matrixlist_all = np.concatenate(matrixlist_all)
    return matrixlist_all, timesteps


# 2DO create a matrixlist which consists of all experiments to normalize the whole set.

if __name__ == "__main__":
    xy = import_all_rooms(False)
    #for key in xy:
    #    print(key, len(xy[key]))

    print(xy['1+'].shape)


    #xy = import_all_rooms(False)









