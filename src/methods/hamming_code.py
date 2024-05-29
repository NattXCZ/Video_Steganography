import cv2 as cv

import numpy as np
from src.utils import w_binary as bnr
from src.utils import video_processing as vid_utils
from src.utils import bcolors as color

import random
#A Highly Secure Video Steganography using Hamming Code (7, 4)

#have three keys, shared between sender and receiver:
#   key_chaotic: Reposition pixels in Y, U, V, and the secret message into a random position, which makes the data chaotic
#
#   key_mess1 and key_mess2:
#                           Used to pick the random rows and columns respectively in each chaotic Y, U and V component.
#

#prvnich 20 prvku zpravy [0 1 0 0 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 1]

# Generator matrix
G = np.array([[1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1]])


H_transposed = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [0, 1, 1],
                    [1, 1, 1],
                    [1, 0, 1]])

max_bits_per_frame = 22500    #maximum of bits saved in one frame (22500)

xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) #random 7-bit value or number 3 (ale to nevychazi, cisla sou osmbit)




def hamming_encode(orig_video_path, message_path, shift_key, col_key, row_key):
    
    
    #*1 + 2)Convert the video stream into individual frames. Separate each frame into its Y (luma), U (chrominance), and V (chrominance) components.
    vid_properties = vid_utils.video_to_yuv_frames(orig_video_path)
    


    #max amount of embedded pixels is 22% of all the pixels in frame (4 bits of message per one pixel)
    max_codew_p_frame = int(vid_properties["height"] * vid_properties["width"] + 0.22)
    #|TODO: co když práva bude krátká a nevyjde na každý frame??
    
    
    
    
    
    #*3)Shift the position of all pixels in the Y, U, and V components by a specific key.
    
    #*4)Convert the message into a one-dimensional array, and then shift the entire message by a shift_key.
    message = bnr.file_to_binary_1D_arr(message_path)
    message = fill_end_zeros(np.roll(message, shift_key))
    #message = bnr.add_EOT_sequence(message)
    
    
    #count how much frames will be stored in each frame
    codew_p_frame, codew_p_last_frame =  distribution_of_bits_between_frames(len(message),vid_properties["frames"])
    
    
    if codew_p_frame > max_codew_p_frame:
        print("[INFO] Message is too large to embed")
        return
    
    actual_max_codew = codew_p_frame
    
    #*5)Encode each 4-bit block of the message using a Hamming (7, 4) code.
    row = 0
    col = 0
    embedded_codewords_per_frame = 0
    curr_frame = 1
    
    for i in range(0, len(message), 4):
        four_bits = message[i:i+4]

        
        #create codeword (Hamming code (7,4))
        codeword = (np.dot(four_bits, G) % 2)



        #*6)XOR the resulting 7-bit encoded data (4 message bits + 3 parity bits) with a random 7-bit value using a key.
        codeword = codeword ^ xor_key 



        #*7)Embed the resulting 7 bits into one pixel of the YUV components (3 bits in Y, 2 bits in U, and 2 bits in V).

        #initialize new frames
        if embedded_codewords_per_frame == 0:
            y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
            u_component_path = f"./tmp/U/frame_{curr_frame}.png"
            v_component_path = f"./tmp/V/frame_{curr_frame}.png"

            y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
            u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
            v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)
            
            y_frame_shuffled = seeded_shuffle_image(y_frame, shift_key)
            u_frame_shuffled = seeded_shuffle_image(u_frame, shift_key)
            v_frame_shuffled = seeded_shuffle_image(v_frame, shift_key)
            
            row = 0
            col = 0
            

        #FIXME:oboje bin() i format vraci string - nemelo by delat problemy
        """y_binary_value = bin(y_frame_shuffled[row, col])
        u_binary_value = bin(u_frame_shuffled[row, col])
        v_binary_value = bin(v_frame_shuffled[row, col])"""
        
        y_binary_value = format(y_frame_shuffled[row, col], '#010b')
        u_binary_value = format(u_frame_shuffled[row, col], '#010b')
        v_binary_value = format(v_frame_shuffled[row, col], '#010b')


        #embed bits of codeword
        y_frame_shuffled[row, col] = int(y_binary_value[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        u_frame_shuffled[row, col] = int(u_binary_value[:-2] + ''.join(str(bit) for bit in codeword[3:5]), 2)
        v_frame_shuffled[row, col] = int(v_binary_value[:-2] + ''.join(str(bit) for bit in codeword[5:]), 2)
        
        
        """#FIXME:
        y_frame_shuffled[row, col] = 255
        u_frame_shuffled[row, col] = 0
        v_frame_shuffled[row, col] = 0"""







        embedded_codewords_per_frame += 1    #zvyší se pocet vlozenych kodovych slov
        # Update row and col after processing a pixel
        col += 1
        if col >= int(vid_properties["width"]):  # Reached end of current row (in enery YUV frame it is a same value)
            col = 0
            row += 1
            
        if embedded_codewords_per_frame >= actual_max_codew:  # Pokud je pozice větší než délka zprávy, přejdi na další snímek
            curr_frame += 1
            embedded_codewords_per_frame = 0
            #ulozeni frames
            
            
            
            y_frame = seeded_unshuffle_image(y_frame_shuffled, shift_key)
            u_frame = seeded_unshuffle_image(u_frame_shuffled, shift_key)
            v_frame = seeded_unshuffle_image(v_frame_shuffled, shift_key)
            
            cv.imwrite(y_component_path, y_frame)
            cv.imwrite(u_component_path, u_frame)
            cv.imwrite(v_component_path, v_frame)
                
            if curr_frame == vid_properties["frames"]:
                actual_max_codew = codew_p_last_frame

        

    #FIXME: proc to je tady tak
    if embedded_codewords_per_frame > 0:
        #ulozeni frames pokud se neulozilo kdyz bylo embedded_codewords_per_frame == 0
        cv.imwrite(y_component_path, y_frame)
        cv.imwrite(u_component_path, u_frame)
        cv.imwrite(v_component_path, v_frame)

    print(f"[INFO] encoded to frames")
    #*8)Shift the positions of all pixels in the YUV components back to their original positions in the frame pixel grid.


    #*9)Reconstruct the video stream by combining the embedded frames.

    vid_utils.reconstruct_video_from_yuv_frames(orig_video_path, vid_properties)  

    print(f"[INFO] embedding finished")
    


    return vid_properties
    
    

def hamming_decode(stego_video_path, shift_key, col_key, row_key,vid_properties,message_len, output_path):
    
 
    #FIXME: aby nevytvarelo furt novej list
    decoded_message = []
    #decoded_message = np.zeros(message_len, dtype = np.uint8)
    
    
    codeword_chaos = np.array([0, 0, 0, 0, 0, 0, 0])
    
    
    decoded_codeword = []   

    # 1) Convert the video stream into frames. Separate each frame into Y, U and V components.

    #FIXME: Zkouška bez tvoření noveho videa
    
    
    #vid_properties = vid_utils.video_to_yuv_frames(stego_video_path)
    
    
    
    codew_p_frame, codew_p_last_frame =  distribution_of_bits_between_frames(message_len,vid_properties["frames"])
    
    actual_max_codew = codew_p_frame

    # 3) Change the position of all pixel values in the three Y, U, and V components by the special key that was used in the embedding process.


    # 4) Obtain the encoded data from the YUV components and XOR with the random number using the same key that was used in the sender side.

    #v cyklu projit secky frames dokud nenajdu končnou sekvenci (az najde prvni ze sekvence, zvysi citatc na 1, esi najde druhy zvysi o jedno vic 
    #pokud neco sekvenci pokazi tak nenajde tak jej da na nulu  aznova jede)
    # v kazdem pixelu snimnku odkoduje slovo a to hned dekoduje a ulozi do odpovedi , asi ty kontrolni sekvence pridat jeste pred kodovanim 



    for curr_frame in range(1, int(vid_properties["frames"]) + 1): #bere od 1 - frames
        embedded_codewords = 0

        #load new frame
        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv.imread(y_component_path, cv.IMREAD_GRAYSCALE)
        u_frame = cv.imread(u_component_path, cv.IMREAD_GRAYSCALE)
        v_frame = cv.imread(v_component_path, cv.IMREAD_GRAYSCALE)
        
        y_frame_shuffled = seeded_shuffle_image(y_frame, shift_key)
        u_frame_shuffled = seeded_shuffle_image(u_frame, shift_key)
        v_frame_shuffled = seeded_shuffle_image(v_frame, shift_key)
        
        
        
        if curr_frame == vid_properties["frames"]:
            actual_max_codew = codew_p_last_frame

        stop_loop = False
        #for row in range(1, int(vid_properties["height"])):
        for row in range(int(vid_properties["height"])):
            if stop_loop:
                break
            
            #for col in range(1, int(vid_properties["width"])):
            for col in range(int(vid_properties["width"])):

                if embedded_codewords >= actual_max_codew:    # esi vlozi tolik slov kolik ma tak se presune na další frame
                    stop_loop = True
                    break
                

                """y_binary_value = bin(y_frame_shuffled[row, col])
                u_binary_value = bin(u_frame_shuffled[row, col])
                v_binary_value = bin(v_frame_shuffled[row, col])"""
                
                y_binary_value = format(y_frame_shuffled[row, col], '#010b')
                u_binary_value = format(u_frame_shuffled[row, col], '#010b')
                v_binary_value = format(v_frame_shuffled[row, col], '#010b')
                
                
                #FIXME:
                """y_frame[row, col] = 0
                u_frame[row, col] = 255
                v_frame[row, col] = 0"""

                
                #TODO: dopsat aby "vymazalo zprávu - tím že tam doplní samý nuly"

                codeword_chaos[0] = y_binary_value[-3]
                codeword_chaos[1] = y_binary_value[-2]
                codeword_chaos[2] = y_binary_value[-1]
    
                codeword_chaos[3] = u_binary_value[-2]
                codeword_chaos[4] = u_binary_value[-1]
    
                codeword_chaos[5] = v_binary_value[-2]
                codeword_chaos[6] = v_binary_value[-1]



                

                codeword = codeword_chaos ^ xor_key           #2 times XOR returns original codeword
                
                decoded_codeword = hamming_decode_codeword(codeword, H_transposed)
                decoded_message.extend(decoded_codeword)
                

                embedded_codewords += 1
                
                
                
        """cv.imwrite(y_component_path, y_frame)
        cv.imwrite(u_component_path, u_frame)
        cv.imwrite(v_component_path, v_frame)"""
        
      
        






    # 6) Reposition the whole message again into the original order.


    # 7) Return converted message (and convert it as a text file or find which file it was)
    # tady je jen azchytnej bod kdy ulozi to co nasel a vrati ze zpravu neslo dekodovat
    message_array = np.array(decoded_message)
    
    
    bnr.binary_1D_arr_to_file(np.roll(message_array, - shift_key), output_path)  #proc tady bylo xor_key


        
    print(f"[INFO] saved decoded message as {output_path}")



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
    return R[-4:]    #TODO proc je tam *-4 kdyz bez minuska necha to stejny


##################
#helper methods
def fill_end_zeros(input_array):
    length = len(input_array)

    if length % 4 == 0:
        return input_array
    else:
        num_zeros = 4 - (length % 4)
        adjusted_array = np.pad(input_array, (0, num_zeros), mode='constant', constant_values=0)

        return adjusted_array
    
def distribution_of_bits_between_frames(len_message, frame_count):
  codew_in_msg = len_message // 4  # Celkový počet slov (děleno 4 bity na slovo)

  codew_p_frame, tail = divmod(codew_in_msg, frame_count)

  return codew_p_frame, codew_p_frame + tail


##############
#shuffle
def seeded_shuffle_image(image, seed):
    rng = random.Random(seed)
    flattened = image.flatten()
    length = len(flattened)
    
    for i in range(length - 1, -1, -1):
        j = rng.randint(0, i)
        flattened[i], flattened[j] = flattened[j], flattened[i]
    
    return flattened.reshape(image.shape)

def seeded_unshuffle_image(image, seed):
    rng = random.Random(seed)
    flattened = image.flatten()
    length = len(flattened)
    
    indices = [rng.randint(0, i) for i in range(length - 1, -1, -1)]
    unshuffle_indices = list(range(length))
    
    for i, j in enumerate(indices[::-1]):
        unshuffle_indices[i], unshuffle_indices[j] = unshuffle_indices[j], unshuffle_indices[i]
    
    unshuffled = np.empty_like(flattened)
    for i, original_index in enumerate(unshuffle_indices):
        unshuffled[i] = flattened[original_index]
    
    return unshuffled.reshape(image.shape)




