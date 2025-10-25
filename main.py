import cv2
import numpy as np
from scipy.fftpack import dct, idct
import os

class DCTSteganography:
    def __init__(self):
        self.quantization_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        self.LENGTH_HEADER_BITS = 32

    def _message_to_binary(self, message):
        message_bytes = message.encode('utf-8')
        return ''.join(format(byte, '08b') for byte in message_bytes)

    def _binary_to_message(self, binary_message):
        byte_values = []
        for i in range(0, len(binary_message), 8):
            byte_segment = binary_message[i:i+8]
            if len(byte_segment) < 8: break
            byte_values.append(int(byte_segment, 2))
        
        message_bytes = bytes(byte_values)
        try:
            return message_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return None

    def _apply_dct(self, image_channel):
        h, w = image_channel.shape
        dct_blocks = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = image_channel[i:i+8, j:j+8].astype(np.float32) - 128
                dct_blocks.append(dct(dct(block.T, norm='ortho').T, norm='ortho'))
        return dct_blocks, h, w

    def _apply_idct(self, dct_blocks, h, w):
        reconstructed_image = np.zeros((h, w))
        block_index = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                reconstructed_block = idct(idct(dct_blocks[block_index].T, norm='ortho').T, norm='ortho')
                reconstructed_image[i:i+8, j:j+8] = reconstructed_block
                block_index += 1
        return reconstructed_image + 128

    def hide_message(self, image_path, message, output_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError("Image not found.")

        h_orig, w_orig, _ = img.shape
        padded_img = self._pad_image(img)
        
        ycbcr_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycbcr_img)

        binary_message = self._message_to_binary(message)
        message_bit_length = len(binary_message)
        
        length_binary = format(message_bit_length, f'0{self.LENGTH_HEADER_BITS}b')
        full_binary_payload = length_binary + binary_message
        
        dct_blocks, h, w = self._apply_dct(y_channel)
        
        # Warning Message Test
        if len(full_binary_payload) > len(dct_blocks):
                    # Calculate the ACTUAL size of the message in bytes using UTF-8, It was Being wronly estimated.
                    message_byte_count = len(message.encode('utf-8'))


                    #Calculate the image's true capacity in bytes.
                    image_capacity_bytes = (len(dct_blocks) - self.LENGTH_HEADER_BITS) // 8
                    error_message = (
                        f"Message is too large for this image.\n\n"
                        f"  Image Capacity:      {image_capacity_bytes} bytes\n"
                        f"  Your Message's Size: {message_byte_count} bytes\n\n"
                        f"Note: Your message size is larger than its character count or maybe because you're using\n"
                        f"special characters and emojis which take up more space(double check)."
                    )
                    raise ValueError(error_message)
        
        payload_index = 0
        for i in range(len(full_binary_payload)):
            quantized_block = np.round(dct_blocks[i] / self.quantization_table).astype(np.int32)
            if int(full_binary_payload[payload_index]) == 0:
                quantized_block[2, 1] &= ~1
            else:
                quantized_block[2, 1] |= 1
            payload_index += 1
            dct_blocks[i] = quantized_block * self.quantization_table
        
        reconstructed_y = self._apply_idct(dct_blocks, h, w)
        reconstructed_y = np.clip(reconstructed_y, 0, 255).astype(np.uint8)
        
        stego_img_ycbcr = cv2.merge([reconstructed_y, cr_channel, cb_channel])
        stego_img_bgr = cv2.cvtColor(stego_img_ycbcr, cv2.COLOR_YCrCb2BGR)
        stego_img_bgr = stego_img_bgr[:h_orig, :w_orig, :]
        
        cv2.imwrite(output_path, stego_img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"\nSUCCESS: Message successfully hidden in '{output_path}'")

    def reveal_message(self, image_path):
        stego_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if stego_img is None: raise FileNotFoundError("Stego image not found.")

        padded_img = self._pad_image(stego_img)
        ycbcr_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2YCrCb)
        y_channel, _, _ = cv2.split(ycbcr_img)

        dct_blocks, _, _ = self._apply_dct(y_channel)
        
        if len(dct_blocks) < self.LENGTH_HEADER_BITS: return None

        length_binary = ""
        for i in range(self.LENGTH_HEADER_BITS):
            quantized_block = np.round(dct_blocks[i] / self.quantization_table).astype(np.int32)
            length_binary += str(quantized_block[2, 1] & 1)
        
        try: message_length = int(length_binary, 2)
        except ValueError: return None
        
        if message_length > len(dct_blocks) - self.LENGTH_HEADER_BITS: return None

        binary_message = ""
        for i in range(message_length):
            block_index = self.LENGTH_HEADER_BITS + i
            quantized_block = np.round(dct_blocks[block_index] / self.quantization_table).astype(np.int32)
            binary_message += str(quantized_block[2, 1] & 1)

        return self._binary_to_message(binary_message)

    def _pad_image(self, img):
        h_orig, w_orig, _ = img.shape
        h_pad = (8 - h_orig % 8) % 8
        w_pad = (8 - w_orig % 8) % 8
        return np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant', constant_values=0)

# Interactive menu.
if __name__ == '__main__':
    steganographer = DCTSteganography()
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        MESSAGE_FILENAME = "message.txt"
        MESSAGE_FILE_PATH = os.path.join(script_dir, MESSAGE_FILENAME)
    except NameError:
        MESSAGE_FILENAME = "message.txt"
        MESSAGE_FILE_PATH = MESSAGE_FILENAME

    while True:
        print("\n--- DCT Steganography Menu ---")
        print(f"1. Hide message from '{MESSAGE_FILENAME}' (Encode)")
        print("2. Reveal a message (Decode)")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            try:
                cover_path = input("Enter path to the cover image (e.g., cover.png): ")
                if not os.path.exists(cover_path): print("ERROR: Cover image not found."); continue
                
                print(f"Attempting to load message from '{MESSAGE_FILE_PATH}'...")
                if not os.path.exists(MESSAGE_FILE_PATH): print(f"ERROR: Message file '{MESSAGE_FILENAME}' not found."); continue

                with open(MESSAGE_FILE_PATH, 'r', encoding='utf-8') as f: message = f.read()
                if not message: print("ERROR: The message file is empty."); continue

                stego_path = input("Enter the output path for the stego-image (e.g., stego.png): ")
                steganographer.hide_message(cover_path, message, stego_path)
            except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
            except Exception as e: print(f"\nAn unexpected error occurred: {e}")
        
        elif choice == '2':
            try:
                stego_path = input("Enter path to the stego-image (e.g., stego.png): ")
                if not os.path.exists(stego_path): print("ERROR: Stego-image not found."); continue
                revealed_message = steganographer.reveal_message(stego_path)
                if revealed_message is not None:
                    print("\n---------------------------------")
                    print("SUCCESS: Revealed message found!")
                    print("---------------------------------")
                    print(revealed_message)
                    print("---------------------------------")
                else:
                    print("\nINFO: No hidden message found or the data is corrupt.")
            except FileNotFoundError as e: print(f"\nERROR: {e}")
            except Exception as e: print(f"\nAn unexpected error occurred: {e}")
            
        elif choice == '3':
            print("Exiting program. Goodbye!"); break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")