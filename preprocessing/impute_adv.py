import numpy as np

def impute_adv(data):

    #get missing channels indices
    # Find indices of rows where all elements are NaN
    nan_row = np.where(np.all(np.isnan(data), axis=1))[0]

    #get signal by itsef without nan
    signal = data[~np.any(np.isnan(data), axis=1)]
    #print("signal shape", signal.shape)

    window_len = 5 # window length 
    total_len = 20 #For 5 minutes = 75000 points
    N = int(total_len/window_len) #number of windows
    channels = data.shape[0]
    imputed_channel = np.empty([])
    
   
    for n in range(2, N-1):
        #calculate correlation matricies for the before and after windows
        pre =np.corrcoef(signal[:, 1+window_len*(n-1):window_len * n])
        pro = np.corrcoef(signal[:, 1+window_len*(n-1):window_len * (n+2)])

        #average the the windows
        weights = (pre + pro)/2
        #print("weight", "\n", weights)
        #get row of weights matrix corresponding to missing row
        weights_row = weights[nan_row, :].ravel()
        #print("signal len", signal.shape[0])
        print("weights row ", weights_row.shape)

        weighted_channels = np.empty((len(weights_row), window_len-1))
        #print("weighted_channels shape", weighted_channels.shape)

        for r in range(channels -1): 
            print("r: ", r)
            #multiply each weight in the weight matrix by its corresponding channel window
            weighted_channel = weights_row[r] * signal[r, 1+ window_len*(n):window_len * (n+1)]

            print("weighted channel", weighted_channel.shape)
            weighted_channels[r, :] = weighted_channel

            #sum the values
            column_sums = np.sum(weighted_channels, axis=0)
            print("column sums" , column_sums)
            #append this to the imputed row
            imputed_channel = np.append(imputed_channel, column_sums)
            print("appending", imputed_channel.shape)

    print("imputed", imputed_channel.shape)
    imputed = imputed_channel[1:len(imputed_channel)]
    print("imputed", imputed.shape)
    data[nan_row, :] = imputed
    print(data)


    return data

#pass in just data, no metadata. Use the reshaped from Sharon 
def run_impute(data):
    num_recordings = data.shape[0]
    for i in range(num_recordings):
        imputed = impute_adv(data[num_recordings, :, : ])
        data[num_recordings, :, : ] = imputed
    return data 


# Create two random 2D arrays filled with random floats
array1 = np.random.rand(3, 20)
array2 = np.random.rand(3, 20)

# Replace one row in array1 with NaN
array1[1, :] = np.nan
array2[1, :] = np.nan

stacked_array = np.stack((array1, array2), axis=0)

# Transpose the array to shape (3, 4, 2)
transposed_array = np.transpose(stacked_array, (1, 2, 0))

run_impute(transposed_array)





