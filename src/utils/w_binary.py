import magic
import numpy as np


def file_to_binary_1D_arr(file_path):
    """ Converts the contents of a file to a 1D numpy array of binary values (0s and 1s)."""
    with open(file_path, 'rb') as file:
        binary_data = file.read()
        binary_array = np.array([int(bit) for byte in binary_data for bit in f"{byte:08b}"], dtype=np.uint8)
        print(f"[INFO] The file '{file_path}' has been successfully converted to a 1D numpy array.")
    return binary_array

def binary_1D_arr_to_file(binary_array, file_path):
    """Converts a 1D numpy array of binary values back to a file."""
    binary_string = ''.join(binary_array.astype(str))
    byte_list = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
    byte_arr = bytearray(byte_list)
    
    with open(file_path, 'wb') as file:
        file.write(byte_arr)
        print(f"[INFO] The binary array has been successfully converted to the file '{file_path}'.")
        
     
def add_EOT_sequence(message):
    sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    new_message = np.concatenate((message, sequence))
    return new_message

def check_EOT_sequence(message):
    sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.array_equal(message[-48:], sequence)


#------------------------------------------------------testing
def iterate_over_4_bits(binary_data):
    """Iterates over the binary data in 4-bit chunks and prints them."""

    for i in range(0, len(binary_data), 4):
        four_bits = binary_data[i:i+4]  # Extract a 4-bit chunk
        print(four_bits, end=" ")  # Print the 4-bit chunk




