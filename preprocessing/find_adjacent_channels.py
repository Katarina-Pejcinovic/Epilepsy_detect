
def find_adjacent_channels(channel):
    
    # bipolar_pairs = [
    #         ('Fp1', 'F7'), ('Fp2', 'F8'), ('F7', 'T3'), ('F8', 'T4'),
    #         ('T3', 'T5'), ('T4', 'T6'), ('T5', 'O1'), ('T6', 'O2'), 
    #         ('A1','T3'), ('T4', 'A2'), ('T3', 'C3'), ('C4', 'T4'),
    #         ('C3', 'Cz'),('Cz', 'C4'), ('Fp1', 'F3'), ('Fp2', 'F4'),
    #         ('F3', 'C3'), ('F4', 'C4'), ('C3', 'P3'), ('C4', 'P4'),
    #         ('P3', 'O1'), ('P4', 'O2'), ('Pz', 'P4'), ('Fz', 'Cz'),
    #         ('Cz', 'Pz'), ('Pz', 'Oz'), ('Fz', 'Fpz'), ('Fpz', 'Fp1'),
    #         ('Fpz', 'Fp2'), ('O1', 'Oz'), ('O2', 'Oz'), 
    #         ('P3', 'Pz'), ('Fz', 'F4'), ('F3', 'Fz')  
    # ]
    
    adjacent_channels_dict = {'Fp1': ['Fpz', 'F7', 'F3', 'Fz']
                           , 'F7': ['Fp1', 'F3', 'T3', 'C3']
                           , 'Fp2': ['Fpz', 'Fz', 'F4', 'F8']
                           , 'F8': ['Fp2', 'F4', 'C4', 'T4']
                           , 'T3': ['A1', 'F7', 'C3', 'T5']
                           , 'T4': ['A2', 'C4', 'F8', 'T6']
                           , 'T5': ['T3', 'C3', 'P3', 'O1']
                           , 'T6': ['C4', 'T4', 'P4', 'O2']
                           , 'O1': ['T5', 'P3', 'Pz', 'Oz']
                           , 'O2': ['P4', 'T6', 'Pz', 'Oz']
                           , 'A1': ['F7', 'T3', 'T5']
                           , 'A2': ['F8', 'T4', 'T6']
                           , 'C3': ['F3', 'T3', 'Cz', 'P3']
                           , 'C4': ['F4', 'T4', 'Cz', 'P4']
                           , 'Cz': ['Fz', 'C3', 'Pz', 'C4']
                           , 'F3': ['Fp1', 'F7', 'Fz', 'C3']
                           , 'F4': ['Fp2', 'F8', 'Fz', 'C4']
                           , 'P3': ['C3', 'T5', 'Pz', 'O1']
                           , 'P4': ['C4', 'Pz', 'T6', 'O2']
                           , 'Pz': ['Cz', 'P3', 'P4', 'Oz']
                           , 'Fz': ['Fpz', 'F3', 'F4', 'Cz']
                           , 'Oz': ['Pz', 'O1', 'O2']
                           , 'Fpz': ['Fp1', 'Fp2', 'Fz']
    }

    try:
        target_channels = adjacent_channels_dict[channel]
    except:
        print('Invalid channel')
        target_channels = []
    
    return target_channels

    


# bipolar_pairs = [
#             ('Fp1', 'F7'), ('Fp2', 'F8'), ('F7', 'T3'), ('F8', 'T4'),
#             ('T3', 'T5'), ('T4', 'T6'), ('T5', 'O1'), ('T6', 'O2'), 
#             ('A1','T3'), ('T4', 'A2'), ('T3', 'C3'), ('C4', 'T4'),
#             ('C3', 'Cz'),('Cz', 'C4'), ('Fp1', 'F3'), ('Fp2', 'F4'),
#             ('F3', 'C3'), ('F4', 'C4'), ('C3', 'P3'), ('C4', 'P4'),
#             ('P3', 'O1'), ('P4', 'O2'), ('Pz', 'P4'), ('Fz', 'Cz'),
#             ('Cz', 'Pz'), ('Pz', 'Oz'), ('Fz', 'Fpz'), ('Fpz', 'Fp1'),
#             ('Fpz', 'Fp2'), ('O1', 'Oz'), ('O2', 'Oz'), 
#             ('P3', 'Pz'), ('Fz', 'F4'), ('F3', 'Fz')  
#     ]

# Find all unique channels in bipolar pairs

# channels = []
# for pairs in bipolar_pairs:
#     channels.append(pairs[0])
#     channels.append(pairs[1])

# unique_list = []
# for x in channels:
#     if x not in unique_list:
#         unique_list.append(x)

# print(len(unique_list))
# print(unique_list)

# import pickle

# adjacent_channels_dict = {'Fp1': ['Fpz', 'F7', 'F3', 'Fz']
#                            , 'F7': ['Fp1', 'F3', 'T3', 'C3']
#                            , 'Fp2': ['Fpz', 'Fz', 'F4', 'F8']
#                            , 'F8': ['Fp2', 'F4', 'C4', 'T4']
#                            , 'T3': ['A1', 'F7', 'C3', 'T5']
#                            , 'T4': ['A2', 'C4', 'F8', 'T6']
#                            , 'T5': ['T3', 'C3', 'P3', 'O1']
#                            , 'T6': ['C4', 'T4', 'P4', 'O2']
#                            , 'O1': ['T5', 'P3', 'Pz', 'Oz']
#                            , 'O2': ['P4', 'T6', 'Pz', 'Oz']
#                            , 'A1': ['F7', 'T3', 'T5']
#                            , 'A2': ['F8', 'T4', 'T6']
#                            , 'C3': ['F3', 'T3', 'Cz', 'P3']
#                            , 'C4': ['F4', 'T4', 'Cz', 'P4']
#                            , 'Cz': ['Fz', 'C3', 'Pz', 'C4']
#                            , 'F3': ['Fp1', 'F7', 'Fz', 'C3']
#                            , 'F4': ['Fp2', 'F8', 'Fz', 'C4']
#                            , 'P3': ['C3', 'T5', 'Pz', 'O1']
#                            , 'P4': ['C4', 'Pz', 'T6', 'O2']
#                            , 'Pz': ['Cz', 'P3', 'P4', 'Oz']
#                            , 'Fz': ['Fpz', 'F3', 'F4', 'Cz']
#                            , 'Oz': ['Pz', 'O1', 'O2']
#                            , 'Fpz': ['Fp1', 'Fp2', 'Fz']
#     }

# with open('data/adjacent_channels_dict.pkl', 'wb') as f:
#     pickle.dump(adjacent_channels_dict, f)

# found = True
# for i in adjacent_channels_dict:
#     chan = adjacent_channels_dict[i]
#     for j in chan:
#         if j not in unique_list:
#             print('Not found: ', j)
#             found = False

# if found:
#     print('All found')

test = find_adjacent_channels('T3')
print('Found: ', test)
print(type(test))
test1 = find_adjacent_channels('T1')
print('Not found: ', test1)