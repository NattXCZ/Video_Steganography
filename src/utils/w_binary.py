import magic
import numpy as np


def read_file_to_binary(file_path):
    """Reads the contents of a file and returns them as binary data."""
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return binary_data

    #retezec na binarni reprezentaci
    #binary_data= int(inputdata, base=2)
    

def determine_file_type(binary_data):
    """Determines the type of file based on its binary data."""
    mime_type = magic.from_buffer(binary_data, mime=True)
    if mime_type.startswith('image'):
        return 'image'
    elif mime_type.startswith('video'):
        return 'video'
    elif mime_type.startswith('text'):
        return 'text'
    else:
        return 'unknown'


#------------------------------------------------------testing
def iterate_over_4_bits(binary_data):
    """Iterates over the binary data in 4-bit chunks and prints them."""

    for i in range(0, len(binary_data), 4):
        four_bits = binary_data[i:i+4]  # Extract a 4-bit chunk
        print(four_bits, end=" ")  # Print the 4-bit chunk


def read_file_to_binary_string(file_path):
  """Načte obsah textového souboru a vrátí binární reprezentaci textu."""
  with open(file_path, "rb") as f:
    byte_data = f.read()
  return "".join(f"{byte:08b}" for byte in byte_data)



def read_file_to_1D(file_path):
  """
  Reads the contents of a file and returns them as a 1D array of zeros and ones.

  Args:
      file_path (str): The path to the file to be read.

  Returns:
      numpy.ndarray: A 1D array of zeros and ones representing the binary data.
  """
  with open(file_path, 'rb') as file:
    binary_data = file.read()

  # Convert each byte to its binary string representation
  byte_strings = [f"{byte:08b}" for byte in binary_data]

  # Join the binary strings and convert them into a 1D array of integers (0 or 1)
  binary_array = np.array([int(bit) for bit in "".join(byte_strings)])

  return binary_array
