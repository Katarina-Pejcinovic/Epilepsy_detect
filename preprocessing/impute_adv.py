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


    return 


#test case 
import numpy as np

# Create two random 2D arrays filled with random floats
array1 = np.random.rand(6, 20)
# Replace one row in array1 with NaN
array1[1, :] = np.nan

#print("array 1", "\n", array1)

impute_adv(array1)





