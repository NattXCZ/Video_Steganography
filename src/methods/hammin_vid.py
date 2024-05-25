import cv2 as cv
#import magic
import os

import numpy as np
from src.utils import w_binary as bnr
from src.utils import video_processing as vid_utils
import cv2

from src.methods import  hamming_code as hmc

#! tohle se me libi !! https://github.com/DLarisa/Dissertation/blob/master/Diserta%C8%9Bie/Algoritm%20Propus%20%2B%20GUI/code_steganography.py

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


#TODO: ošetrit když bude vkladat moc velka data !
max_bits_per_frame = 22500    #maximum of bits saved in one frame


def hamming_encode(orig_video_path, message_path, shift_key):
    """
    Encodes 4-bit data into a 7-bit Hamming codeword.

    Args:
        data_binary: A 4-bit binary data to encode.

    Returns:
        A 7-bit string representing the encoded Hamming codeword.
    """
    #FIXME: jen testing

    xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) #random 7-bit value or number 3 (ale to nevychazi, cisla sou osmbit)

    row_key = 19
    col_key = 4


    max_codew_p_frame = max_bits_per_frame / 4      # xth codeword we are embedding rn
    

    #1)Convert the video stream into individual frames. Separate each frame into its Y (luma), U (chrominance), and V (chrominance) components.
    vid_properties = vid_utils.video_to_yuv_frames(orig_video_path)

    #|TODO:3)Shift the position of all pixels in the Y, U, and V components by a specific key.
    

    #4)Convert the message (a binary image) into a one-dimensional array, and then shift the entire message by a key.
    message = bnr.file_to_binary_1D_arr(message_path)
    tmp_message = np.roll(message, shift_key)
    tmp_message2 = hmc.fill_end_zeros(tmp_message)
    message = bnr.add_EOT_sequence(tmp_message2)
    
    if len(message) >= max_bits_per_frame * vid_properties["frames"]:
        print("[INFO] secrets message is huge")
        print(len(message))
        print(max_bits_per_frame * vid_properties["frames"])
        #TODO: aby nemuselo predtím extrahovat snimky ale zjistilo rovnou FPS
        return

    #5)Encode each 4-bit block of the message using a Hamming (7, 4) code.




    row = 1
    col = 1
    embedded_codewords_per_frame = 0
    curr_frame = 1                  #frame we are embedding to

    print("ZACINA VKLADAT")
    for i in range(0, len(message), 4):
        four_bits = message[i:i+4]

        lennnn = len(message)
        len_curr_mess = len(message[i:i+4])
        #print(lennnn)
        
        #create codeword (Hamming code (7,4))
        codeword = (np.dot(four_bits, G) % 2)

        #6)XOR the resulting 7-bit encoded data (4 message bits + 3 parity bits) with a random 7-bit value using a key.
        codeword = codeword ^ xor_key 

        #7)Embed the resulting 7 bits into one pixel of the YUV components (3 bits in Y, 2 bits in U, and 2 bits in V).

        #initialize new frames
        #FIXME: kouknou jak jsou frames uloženy (s jakymi čisli)
        if embedded_codewords_per_frame == 0:
            y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
            u_component_path = f"./tmp/U/frame_{curr_frame}.png"
            v_component_path = f"./tmp/V/frame_{curr_frame}.png"

            y_frame = cv.imread(y_component_path)# cv.IMREAD_GRAYSCALE)
            u_frame = cv.imread(u_component_path)# cv.IMREAD_GRAYSCALE)
            v_frame = cv.imread(v_component_path)# cv.IMREAD_GRAYSCALE)
            
            print(f"frame {curr_frame}")
            print(f"col = {col}, row = {row}")
            row = 1
            col = 1
            


        y_binary_value = bin(y_frame[row, col][0])
        u_binary_value = bin(u_frame[row, col][0])
        v_binary_value = bin(v_frame[row, col][0])


        #embed bits of codeword
        y_frame[row, col][:] = int(y_binary_value[:-3] + ''.join(str(bit) for bit in codeword[:3]), 2)
        u_frame[row, col][:] = int(u_binary_value[:-2] + ''.join(str(bit) for bit in codeword[3:5]), 2)
        v_frame[row, col][:] = int(v_binary_value[:-2] + ''.join(str(bit) for bit in codeword[5:]), 2)







        embedded_codewords_per_frame += 1    #zvyší se pocet vlozenych kodovych slov
        # Update row and col after processing a pixel
        col += 1
        if col >= int(vid_properties["width"]):  # Reached end of current row (in enery YUV frame it is a same value)
            col = 1
            row += 1

        if embedded_codewords_per_frame >= max_codew_p_frame:  # Pokud je pozice větší než délka zprávy, přejdi na další snímek
            curr_frame += 1
            embedded_codewords_per_frame = 0
            #ulozeni frames
            cv.imwrite(y_component_path, y_frame)
            cv.imwrite(u_component_path, u_frame)
            cv.imwrite(v_component_path, v_frame)
        

            

    if embedded_codewords_per_frame > 0:
        #ulozeni frames pokud se neulozilo kdyz bylo embedded_codewords_per_frame == 0
        cv.imwrite(y_component_path, y_frame)
        cv.imwrite(u_component_path, u_frame)
        cv.imwrite(v_component_path, v_frame)


    #8)Shift the positions of all pixels in the YUV components back to their original positions in the frame pixel grid.


    #9)Reconstruct the video stream by combining the embedded frames.
    print(f"[INFO] RECOSNTRUCTION STARTED")
    vid_utils.reconstruct_video_from_yuv_frames(orig_video_path, vid_properties)   #FIXME: proc tady byla output_path?
    print(f"[INFO] RECOSNTRUCTION FINISHED")
        
        
    print(f"[INFO] embedding finished")
    print(curr_frame)
    print(f"col = {col}, row = {row}")
    return vid_properties


def hamming_decode(orig_video_path, shift_key, output_path = f"./tmp/decoded_message.txt"):
    """
    Decodes a 7-bit Hamming codeword into 4-bit data. This function also corrects single-bit errors in the codeword, if detected.

    Args:
        codeword: A 7-bit string representing the Hamming codeword to decode.

    Returns:
        A 4-bit string representing the decoded data.
    """
    
    #FIXME: potom odelat: jen testing esi je problem v zastaveni nebo kodovani
    message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
    orig_msg = bnr.file_to_binary_1D_arr(message_path)
    
    
    
    xor_key = np.array([1, 1, 1, 0, 0, 1, 1]) #random 7-bit value
    row_key = 19
    col_key = 4



    max_codew_p_frame = max_bits_per_frame / 4      # xth codeword we are embedding rn




    decoded_message = []
    decoded_codeword = ""    #? proč je tady string?
    # 1) Convert the video stream into frames. Separate each frame into Y, U and V components.

    #FIXME: Zkouška bez tvoření noveho videa
    vid_properties = vid_utils.video_to_yuv_frames(orig_video_path)

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
        embedded_codewords = 0

        #load new frame
        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv.imread(y_component_path)# cv.IMREAD_GRAYSCALE)
        u_frame = cv.imread(u_component_path)# cv.IMREAD_GRAYSCALE)
        v_frame = cv.imread(v_component_path)# cv.IMREAD_GRAYSCALE)

        stop_loop = False
        for row in range(1, int(vid_properties["height"])):
            if stop_loop:
                break

            for col in range(1, int(vid_properties["width"])):

                if embedded_codewords > max_codew_p_frame:    # esi vlozi tolik slov kolik ma tak se presune na další frame
                    stop_loop = True
                    break
                

                y_binary_value = bin(y_frame[row, col][0])
                u_binary_value = bin(u_frame[row, col][0])
                v_binary_value = bin(v_frame[row, col][0])
                
                #TODO: dopsat aby "vymazalo zprávu - tím že tam doplní samý nuly"


                codeword_chaos = np.array([int(bit) for bit in y_binary_value[-3:] + u_binary_value[-2:] + v_binary_value[-2:]])

                codeword = codeword_chaos ^ xor_key           #2 times XOR returns original codeword
                
                decoded_codeword = hmc.hamming_decode_codeword(codeword, H_transposed)
                decoded_message.extend(decoded_codeword)

                if np.array_equal(decoded_codeword, arr_ones):
                    eot_1 += 1
                elif np.array_equal(decoded_codeword, arr_zeros):
                    eot_0 += 1
                else:
                    eot_0 = 0
                    eot_1 = 0



                if eot_1 == 6 and eot_0 == 6:
                    if bnr.check_EOT_sequence(decoded_message):
                        message_array= np.array(decoded_message)[:-48]
                        bnr.binary_1D_arr_to_file(np.roll(message_array, - shift_key), output_path)
                        print(f"[INFO] embedding finished")
                        

                        print(np.roll(message_array, - shift_key)[:20])
                        print(orig_msg[:20])
                        
                        return
                    else:
                        eot_0 = 0
                        eot_1 = 0


                embedded_codewords += 1
                
                
        






    # 6) Reposition the whole message again into the original order.


    # 7) Return converted message (and convert it as a text file or find which file it was)
    # tady je jen azchytnej bod kdy ulozi to co nasel a vrati ze zpravu neslo dekodovat
    message_array= np.array(decoded_message)[:-48]
    bnr.binary_1D_arr_to_file(np.roll(message_array, - xor_key), output_path)
    
    print(f"frame {curr_frame}")
    print(f"col = {col}, row = {row}")
    print(np.roll(message_array, - shift_key)[:20])
    print(orig_msg[:20])
    
    print(f"[INFO] could not decode secret message")







#!-------------------------------------------------------------testing

def hamming_decode_test(orig_video_path, shift_key, vid_properties, output_path = f"./tmp/decoded_message.txt"):
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



    max_codew_p_frame = max_bits_per_frame / 4      # xth codeword we are embedding rn




    decoded_message = []
    decoded_codeword = ""    #? proč je tady string?
    # 1) Convert the video stream into frames. Separate each frame into Y, U and V components.

    #FIXME: Zkouška bez tvoření noveho videa
    #vid_properties = vid_utils.video_to_yuv_frames(orig_video_path)

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
        embedded_codewords = 0

        #load new frame
        y_component_path = f"./tmp/Y/frame_{curr_frame}.png"
        u_component_path = f"./tmp/U/frame_{curr_frame}.png"
        v_component_path = f"./tmp/V/frame_{curr_frame}.png"

        y_frame = cv.imread(y_component_path)# cv.IMREAD_GRAYSCALE)
        u_frame = cv.imread(u_component_path)# cv.IMREAD_GRAYSCALE)
        v_frame = cv.imread(v_component_path)# cv.IMREAD_GRAYSCALE)

        stop_loop = False
        for row in range(1, int(vid_properties["height"])):
            if stop_loop:
                break

            for col in range(1, int(vid_properties["width"])):

                if embedded_codewords > max_codew_p_frame:    # esi vlozi tolik slov kolik ma tak se presune na další frame
                    stop_loop = True
                    break
                

                y_binary_value = bin(y_frame[row, col][0])
                u_binary_value = bin(u_frame[row, col][0])
                v_binary_value = bin(v_frame[row, col][0])
                
                #TODO: dopsat aby "vymazalo zprávu - tím že tam doplní samý nuly"


                codeword_chaos = np.array([int(bit) for bit in y_binary_value[-3:] + u_binary_value[-2:] + v_binary_value[-2:]])

                codeword = codeword_chaos ^ xor_key           #2 times XOR returns original codeword
                
                decoded_codeword = hmc.hamming_decode_codeword(codeword, H_transposed)
                decoded_message.extend(decoded_codeword)

                if np.array_equal(decoded_codeword, arr_ones):
                    eot_1 += 1
                elif np.array_equal(decoded_codeword, arr_zeros):
                    eot_0 += 1
                else:
                    eot_0 = 0
                    eot_1 = 0



                if eot_1 == 6 and eot_0 == 6:
                    if bnr.check_EOT_sequence(decoded_message):
                        message_array= np.array(decoded_message)[:-48]
                        bnr.binary_1D_arr_to_file(np.roll(message_array, - shift_key), output_path)
                        
                        message_path = r"C:\Users\natal\OneDrive\Desktop\data__test\input\carousel-horse.png"
                        orig_message = bnr.file_to_binary_1D_arr(message_path)
                        print(f"length original {len(orig_message)}, decoded_mess = {len(message_array)}")
                        print(f"[INFO] embedding finished")
                        return
                    else:
                        eot_0 = 0
                        eot_1 = 0


                embedded_codewords += 1
                

        
        print(f"frame {curr_frame}")
        print(f"col = {col}, row = {row}")





    # 6) Reposition the whole message again into the original order.


    # 7) Return converted message (and convert it as a text file or find which file it was)
    # tady je jen azchytnej bod kdy ulozi to co nasel a vrati ze zpravu neslo dekodovat
    message_array= np.array(decoded_message)[:-48]
    bnr.binary_1D_arr_to_file(np.roll(message_array, - xor_key), output_path)
    print(f"[INFO] could not decode secret message")



