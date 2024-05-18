import cv2 as cv
import magic
import os

import numpy as np
from src.utils import w_binary as binary
from src.utils import video_processing as vid_utils

#A Highly Secure Video Steganography using Hamming Code (7, 4)

#have three keys, shared between sender and receiver:
#   key_chaotic: Reposition pixels in Y, U, V, and the secret message into a random position, which makes the data chaotic
#
#   key_mess1 and key_mess2:
#                           Used to pick the random rows and columns respectively in each chaotic Y, U and V component.
#

#prvnich 20 prvku zpravy [0 1 0 0 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 1]


#UPRAV je to TODO:

def hamming_encode(orig_video_path, message_path, output_path):
    """
    Encodes 4-bit data into a 7-bit Hamming codeword.

    Args:
        data_binary: A 4-bit binary data to encode.

    Returns:
        A 7-bit string representing the encoded Hamming codeword.
    """
    xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) #random 7-bit value or number 3 (ale to nevychazi, cisla sou osmbit)

    row_key = 19
    col_key = 4

    max_bits_per_frame = 22500    #maximum of bits saved in one frame


    max_codew_p_frame = max_bits_per_frame / 4      # xth codeword we are embedding rn



    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    orig_video_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\horse3s.mp4"
    output_path = f"./tmp/output.mp4"

    #1)Convert the video stream into individual frames. Separate each frame into its Y (luma), U (chrominance), and V (chrominance) components.
    vid_properties = vid_utils.video_to_yuv_frames(orig_video_path)

    #3)Shift the position of all pixels in the Y, U, and V components by a specific key.
    

    #4)Convert the message (a binary image) into a one-dimensional array, and then shift the entire message by a key.
    message = binary.read_file_to_1D(message_path)
    temp_mess = add_EOT_sequence(message)

    #FIXME: tady se rotuje o "key"
    message = np.roll(temp_mess, xor_key)

    #FIXME: xx
    print(message[:20])
    print(f"length = {len(message)}")
    #5)Encode each 4-bit block of the message using a Hamming (7, 4) code.


    # Generator matrix
    G = np.array([[1, 1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1]])
    
    #indexing from zero
    row = 0
    col = 0
    embedded_codewords = 0
    curr_frame = 1                  #frame we are embedding to


    #FIXME: tohle je navic (pokud bude mit zpráva celou délku nedělitelnou 4, tak na její konec dodáme nuly a až potom dodáme EOT sekvenci)
    message = fill_end_zeros(message)

    for i in range(0, len(message), 4):
        four_bits = message[i:i+4]

        #TODO: tohle bylo puvodne
        #four_bits = fill_zeros(message[i:i+4], 4)
       

        #create codeword (Hamming code (7,4))
        codeword = (np.dot(four_bits, G) % 2)

        #6)XOR the resulting 7-bit encoded data (4 message bits + 3 parity bits) with a random 7-bit value using a key.
        codeword = codeword ^ xor_key 

        #7)Embed the resulting 7 bits into one pixel of the YUV components (3 bits in Y, 2 bits in U, and 2 bits in V).

        if embedded_codewords == 0:
            y_component_path = f"./tmp/Y/frame_{curr_frame:04d}.bmp"
            u_component_path = f"./tmp/U/frame_{curr_frame:04d}.bmp"
            v_component_path = f"./tmp/V/frame_{curr_frame:04d}.bmp"

            y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
            u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
            v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)

        #TODO: cyklus pro vice pixelu v jednom obrazku ???
        #FIXME: Y : 0b11101010   234
        #       U : 0b1111011   123      !!!neukazuje první nulové bity !!!!!! (ale to očividně nevadi)
        #       V : 0b10000100   132
            


        y_binary_value = bin(y_frame[row, col])
        u_binary_value = bin(u_frame[row, col])
        v_binary_value = bin(v_frame[row, col])


        #embed bits of codeword
        y_frame[row, col] = int(y_binary_value[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        u_frame[row, col] = int(u_binary_value[:-2] + ''.join(str(bit) for bit in codeword[3:5]), 2)
        v_frame[row, col] = int(v_binary_value[:-2] + ''.join(str(bit) for bit in codeword[5:]), 2)





        #save changes in current yuv frames
        cv.imwrite(y_component_path, y_frame)
        cv.imwrite(u_component_path, u_frame)
        cv.imwrite(v_component_path, v_frame)

        embedded_codewords += 1    #zvyší se pocet vlozenych kodovych slov
        # Update row and col after processing a pixel
        col += 1
        if col >= int(vid_properties["width"]):  # Reached end of current row (in enery YUV frame it is a same value)

            col = 0
            row += 1
        print(f"embedded codeords =  {embedded_codewords}, col = {col}, row = {row}, frame = {curr_frame}, message length = {len(message)}")
        if embedded_codewords >= max_codew_p_frame:  # Pokud je pozice větší než délka zprávy, přejdi na další snímek
            print(f"Last pixel for embedding col = {col}, row = {row}, frame = {curr_frame}")
            curr_frame += 1
            embedded_codewords = 0
            




    #8)Shift the positions of all pixels in the YUV components back to their original positions in the frame pixel grid.


    #9)Reconstruct the video stream by combining the embedded frames.
    vid_utils.from_yuv_reconstruct_vid_file(output_path, vid_properties)

    #FIXME: nebude vracet nic, maximálně vytiskne zprávu že je vše dobře provedeno 
    print(message[:48])
    return vid_properties

def hamming_decode(orig_video_path, output_path, properties):
    """
    Decodes a 7-bit Hamming codeword into 4-bit data. This function also corrects single-bit errors in the codeword, if detected.

    Args:
        codeword: A 7-bit string representing the Hamming codeword to decode.

    Returns:
        A 4-bit string representing the decoded data.
    """
    xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) #random 7-bit value
    row_key = 19
    col_key = 4
    max_bits_per_frame = 22500    #maximum of bits saved in one frame
    #curr_frame = 1                  #frame we are embedding to

    max_codew_p_frame = max_bits_per_frame / 4      # xth codeword we are embedding rn
    #row = 1
    #col = 1



    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\secret_text.txt"
    orig_video_path = r"C:\Users\natal\OneDrive\Desktop\data__test\output\output.mp4"
    output_path = f"./tmp/output.txt"

    decoded_message = []
    decoded_codeword = ""
    # 1) Convert the video stream into frames. Separate each frame into Y, U and V components.

    #FIXME: Zkouška bez tvoření noveho videa
    #vid_properties = vid_utils.video_to_yuv_frames(orig_video_path)
    vid_properties = properties

    # 3) Change the position of all pixel values in the three Y, U, and V components by the special key that was used in the embedding process.


    # 4) Obtain the encoded data from the YUV components and XOR with the random number using the same key that was used in the sender side.

    #v cyklu projit secky frames dokud nenajdu končnou sekvenci (az najde prvni ze sekvence, zvysi citatc na 1, esi najde druhy zvysi o jedno vic 
    #pokud neco sekvenci pokazi tak nenajde tak jej da na nulu  aznova jede)
    # v kazdem pixelu snimnku odkoduje slovo a to hned dekoduje a ulozi do odpovedi , asi ty kontrolni sekvence pridat jeste pred kodovanim 

    arr_ones = np.array([1,1,1,1])
    arr_zeros = np.array([0,0,0,0])
    eot_1 = 0
    eot_0 = 0


    for curr_frame in range(1, int(vid_properties["frames"]) + 1): 
        curr_position = 0

        #load new frame
        y_component_path = f"./tmp/Y/frame_{curr_frame:04d}.bmp"
        u_component_path = f"./tmp/U/frame_{curr_frame:04d}.bmp"
        v_component_path = f"./tmp/V/frame_{curr_frame:04d}.bmp"

        y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
        u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
        v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)

        stop_loop = False
        for row in range(int(vid_properties["height"])):
            if stop_loop:
                break

            for col in range(int(vid_properties["width"])):

                if curr_position > max_codew_p_frame:    # esi vlozi tolik slov kolik ma tak se presune na další frame
                    stop_loop = True
                    break
                

                y_binary_value = bin(y_frame[row, col])
                u_binary_value = bin(u_frame[row, col])
                v_binary_value = bin(v_frame[row, col])


                codeword_chaos = np.array([int(bit) for bit in y_binary_value[-3:] + u_binary_value[-2:] + v_binary_value[-2:]])

                codeword = codeword_chaos ^ xor_key           #2 times XOR returns original codeword
                
                H_transposed = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1],
                                     [1, 1, 0],
                                     [0, 1, 1],
                                     [1, 1, 1],
                                     [1, 0, 1]])
                decoded_codeword = hamming_decode_codeword(codeword, H_transposed)
                decoded_message.extend(decoded_codeword)

                if np.array_equal(decoded_codeword, arr_ones):
                    eot_1 += 1
                    #print(f"EOT_0 = {eot_0}, EOT_1 = {eot_1}")
                elif np.array_equal(decoded_codeword, arr_zeros):
                    eot_0 += 1
                    #print(f"EOT_0 = {eot_0}, EOT_1 = {eot_1}")
                else:
                    eot_0 = 0
                    eot_1 = 0



                if eot_1 == 6 and eot_0 == 6:
                    if check_EOT_sequence(decoded_message):
                        print("HEEREEEEE")
                        return decoded_message


                curr_position += 1

    




    #codeword = codeword ^ xor_key           #2 times XOR returns original codeword
    """
    # 5) Decode 4 bits of the message by the Hamming decoder. 

    H_transposed = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 1, 0],
                         [0, 1, 1],
                         [1, 1, 1],
                         [1, 0, 1]])
    
    decoded_message.extend(hamming_decode_codeword(codeword, H_transposed))
    
        #Z = [(element % 2) for element in list(np.dot(codeword, H_transposed))]
    Z = (np.dot(codeword, H_transposed) % 2)
    R = codeword

    #find row representing a error
    index = -1
    for i, H_row in enumerate(H_transposed):
        if np.all(Z == H_row):            #Z == row:
         index = i      #indexing from 0
    
    #change bit on index (if there was an error)
    if index > -1:
        R[index] = 1 - R[index]

    #return first four bits
    decoded_message.extend(R[-4:])"""



    # 6) Reposition the whole message again into the original order.


    # 7) Return converted message (and convert it as a text file or find which file it was)
    print("TUTUTU")
    return decoded_message



def add_EOT_sequence(message):
    sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    new_message = np.concatenate((message, sequence))
    return new_message

def check_EOT_sequence(message):
    sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.array_equal(message[-48:], message)


def hamming_decode_codeword(codeword, H_transposed):

    #Z = [(element % 2) for element in list(np.dot(codeword, H_transposed))]
    Z = (np.dot(codeword, H_transposed) % 2)
    R = codeword

    #find row representing a error
    index = -1
    for i, H_row in enumerate(H_transposed):
        if np.all(Z == H_row):            #Z == row:
         index = i      #indexing from 0
    
    #change bit on index (if there was an error)
    if index > -1:
        R[index] = 1 - R[index]

    #return first four bits
    return R[-4:]    #TODO proc je tam *-4 kdyz ebz minuska necha to stejny


def fill_end_zeros(input_array):
    length = len(input_array)

    if length % 4 == 0:
        return input_array
    else:
        num_zeros = 4 - (length % 4)
        adjusted_array = np.pad(input_array, (0, num_zeros), mode='constant', constant_values=0)

        return adjusted_array
    
def shuffle_frames(key):
    pass

def unshuffle_frames(key):
    pass
#___________________________________________________________________________________________________________________________________________
def fill_zeros(arr, length):

    if len(arr) < length:
        rslt_arr = np.pad(arr, (0, length - len(arr)), mode='constant', constant_values=0)
        return rslt_arr
    else:
        return arr
    

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





def embed_into_frame():
   data_p_frame =  1             #data per frame (max 23 040 codewords)
