def test_cut_segments():
    # Generate sample data
    data_list = [
        [np.random.rand(26, 300000), np.random.rand(26, 100000), np.random.rand(26, 50000)],
        [np.random.rand(26, 200000), np.random.rand(26, 250000), np.random.rand(26, 160000)],
        [np.random.rand(26, 180000), np.random.rand(26, 210000), np.random.rand(26, 300000)],
        [np.random.rand(26, 120000), np.random.rand(26, 50000), np.random.rand(26, 190000)],
    ]

    label_list = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
        np.array([10, 11, 12]),
    ]

    patientID_list = [
        np.array([101, 102, 103]),
        np.array([104, 105, 106]),
        np.array([107, 108, 109]),
        np.array([110, 111, 112]),
    ]

    # Call the function
    result_4d_array, label_result, patientID_result = cut_segments(data_list, label_list, patientID_list)

    # Confirm dimensions and sizes
    assert len(result_4d_array) == 4
    print(f"The resulting list should contain 4 3D arrays. Here's how many it has: {len(result_4d_array)}")

    for i in range(len(result_4d_array)):
        assert result_4d_array[i].shape[1] == 26
        print(f"Channels should be 26 in 3D array {i + 1}. Here's how many it has: {result_4d_array[i].shape[1]}")
        assert result_4d_array[i].shape[2] == 150000
        print(f"Segments should be of length 150000 in 3D array {i + 1}. Here's the length: {result_4d_array[i].shape[2]}")

        # Confirm label and patient ID list sizes match the first dimension of 3D arrays
        num_segments = result_4d_array[i].shape[0]
        assert len(label_result[i]) == num_segments
        print(f"3D array {i + 1} has {num_segments} segments, so the size of the label list should be {num_segments}, and it is: {len(label_result[i])}")
        assert len(patientID_result[i]) == num_segments
        print(f"3D array {i + 1} has {num_segments} segments, so the size of the patient ID list should be {num_segments}, and it is: {len(patientID_result[i])}")

    print("All tests passed!")