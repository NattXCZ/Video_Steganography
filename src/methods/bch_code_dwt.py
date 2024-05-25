
def hamming_encode1(four_bits):
    """
    Encodes 4-bit data into a 7-bit Hamming codeword.

    Args:
        data_binary: A 4-bit binary data to encode.

    Returns:
        A 7-bit string representing the encoded Hamming codeword.
    """
    chaos_key = 0
    row_key = 0
    col_key = 0
    curr_frame = 0 

    # Generator matrix
    G = np.array([[1, 1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1]])

    #create codeword
    return [(element % 2) for element in list(np.dot(four_bits, G))]


def hamming_decode1(codeword):
    """
    Decodes a 7-bit Hamming codeword into 4-bit data. This function also corrects single-bit errors in the codeword, if detected.

    Args:
        codeword: A 7-bit string representing the Hamming codeword to decode.

    Returns:
        A 4-bit string representing the decoded data.
    """

    H_transposed = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 1, 0],
                         [0, 1, 1],
                         [1, 1, 1],
                         [1, 0, 1]])
    
    Z = [(element % 2) for element in list(np.dot(codeword, H_transposed))]
    R = codeword

    #find row representing a error
    index = -1
    for i, row in enumerate(H_transposed):
        if np.all(Z == row):            #Z == row:
         index = i      #indexing from 0
    
    #change bit on index (if there was an error)
    if index > -1:
        R[index] = 1 - R[index]

    #return first four bits
    return R[-4:]



