import pandas as pd
import os

# takes the raw .out types and make it to .csv tables


# returns the total number of lines, number of gridpoints
def file_info(xyz_data):
    names = []
    with open(xyz_data) as f:
        for enu, line in enumerate(f, 1):

            if enu==1:
                coord_list(line.strip().split(' '), names, 0)
                names = [ int(i) for i in names ]
    return names

#converts the rows into a list with single column and appends the list till
#the end of a single block (eg. X-Coordinate)
def coord_list(num, coord, nol):
    for n in num:
        if n != '':
            #print("NOL:", nol)
            coord.append(n)
            nol = nol + 1

# routine to convert the rows of data into a single list for .xyz file
def get_gridpoints(xyz_data, position, nol):
    coord = []
    with open(xyz_data) as f:  # gives out the first line as a string
        next(f)
        for enu, line in enumerate(f, 1):
            if enu>=position:
                coord_list(line.strip().split(' '), coord, nol)
                if enu == nol:
                    break
    return coord

# routine to convert .out file into a list of temp values
def get_datapoints(file):
    l = []
    with open(file, 'r') as handle:
        for line in handle:
            for value in line.split():
                l.append(value)

    l = l[3:]
    return l




n_time = 600 #declare number of simulated timesteps

for dirname in os.listdir('data/small_steps'):
    if dirname != 'Mesh_files': # look into all folder besides Mesh_files
        print('data/small_steps'+'/'+ dirname)

        pre_file = 'data/small_steps\\'+ dirname

        df_list = []
        # get the following for every block
        for i_room in range(4):

            # assigning file names to variables

            room_name = '\\Room_' + str(i_room+1)
            xyz_data = 'data/new/Mesh_files\\' + room_name+'.xyz'

            #datatext = 'Room_2_5_0.q'


            no_col = 6 # number of columns in the .xyz files
            xdir = 0
            ydir = 0
            zdir = 0
            names = [] # list with total number of grids in each direction

            names = file_info(xyz_data)



            gridpoints = round(names[0]*names[1]*names[2]) #calculates the total number of gridpoints
            print("total gridpoints in: {} = {}".format(xyz_data,gridpoints))


            # declaration of variables that are to be used.
            x_coord, y_coord, z_coord, data, var = [], [], [], [], []
            xyz_grid = pd.DataFrame(data)
            temp_list = pd.DataFrame(data)

            end_line = round(gridpoints/no_col) # determines the end point of each block

            #print("Gridpoints", gridpoints)
            #print("End Line", end_line)
            #print("No. of lines:", total_coord, 0*end_line+1, 1*end_line+1, 2*end_line+1, 3*end_line+1, (3+1)*end_line )


            var = ['X','Y','Z','iBlank'] # column names for the .xyz file



            # main routine that loops over all the blocks (here number of blocks = 4)
            # #blocks = x-coordinates, y-coordinates, z-coordinates, iBlanks
            for n in range(0,4):
                nol = 0
                start_line = n*end_line+1
                dummy1 = get_gridpoints(xyz_data, start_line, (n+1)*end_line)
                xyz_grid[var[n]] = dummy1




            dataset_t0 = []

            var_t = []
            for i in range(n_time):
                var_t.append(f't_{i+1}')

            print(var_t)


            # column names for the .out file

            # main routine loops over all the temperature data files (here number of temperature files = 60)
            # each file with a 1 second time interval
            # 60 files * 1 seconds = 60 seconds of simulation
            for i in range (n_time):
                # try to get datapoints from certain file, but sometimes the name is different so we need the except
                try:
                    dummy = get_datapoints(pre_file + room_name+'_'+str((i+1))+'_0.out')

                except:
                    dummy = get_datapoints(pre_file + room_name+'_'+str((i+1))+'_1.out') # different name
                    temp_list[var_t[i]] = dummy

                else:
                    temp_list[var_t[i]] = dummy


            # combining grid points and temperature data points into single data frame
            df = []
            df = pd.concat([xyz_grid, temp_list], axis =1)
            print (df.shape)
            df_list.append(df)
            #print(df)
            #df.to_csv(pre_file + room_name + '.csv')

        experiment_df = pd.concat(df_list, ignore_index = True)
        experiment_df.to_csv(pre_file+"/experiment_df.csv")
        print("experiment_df shape = " + str(experiment_df.shape))



    ###########Notes#
    # xyz_grid - data frame for grid points
    # temp_list - data frame for temperature data points including all time steps
    # df - single data frame with shape (gridpoints, 16)
    # 3 coordinates, 1 iBLANK, 12 temperature data points = 16 columns
    # gridpoints computed automatically from the input file
    # inputs = 1> .xyz file name at Line36,
    #          2> .q file name at Line108 (inside for loop)






