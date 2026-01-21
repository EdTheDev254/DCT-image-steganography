import cv2
import numpy as np
from scipy.fftpack import dct, idct
import os

##UI Imports
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox


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


class StegoAPP(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("DCT Stego APP")
        self.geometry("700x500")
        self.resizable(False, False)

        #init
        self.processor = DCTSteganography()

        self.tab_view = ctk.CTkTabview(self, width=650, height=450)
        self.tab_view.pack(padx=20, pady=20)

        self.tab_hide = self.tab_view.add("Hide Message")
        self.tab_reveal = self.tab_view.add("Reveal Message")

        #placeholder
        # label_hide = ctk.CTkLabel(self.tab_hide, text="Hide Message UI", font=("Arial", 20))
        # label_hide.pack(pady=100)

        # label_reveal = ctk.CTkLabel(self.tab_reveal, text="Reveal Message UI", font=("Arial", 20))
        # label_reveal.pack(pady=100)


        # UI Var(s)
        self.hide_cover_path = ctk.StringVar()
        self.hide_output_path = ctk.StringVar()
        self.reveal_stego_path = ctk.StringVar()

        self.setup_hide_ui()
        self.setup_reveal_ui()


    def setup_hide_ui(self):
        frame_cover = ctk.CTkFrame(self.tab_hide)
        frame_cover.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(frame_cover, text="Cover Image:").pack(side="left", padx=10)
        self.entry_cover = ctk.CTkEntry(frame_cover, textvariable=self.hide_cover_path, width=350, placeholder_text="Select an image...")
        self.entry_cover.pack(side="left", padx=10)
        
        btn_browse_cover = ctk.CTkButton(frame_cover, text="Browse", width=80, command=self.browse_cover_image)
        btn_browse_cover.pack(side="left", padx=10)


        lbl_msg = ctk.CTkLabel(self.tab_hide, text="Secret Message:", anchor="w")
        lbl_msg.pack(fill="x", padx=20, pady=(10, 0))

        self.txt_message = ctk.CTkTextbox(self.tab_hide, height=150)
        self.txt_message.pack(fill="x", padx=20, pady=5)

        frame_out = ctk.CTkFrame(self.tab_hide)
        frame_out.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(frame_out, text="Save Output:").pack(side="left", padx=10)
        self.entry_output = ctk.CTkEntry(frame_out, textvariable=self.hide_output_path, width=350, placeholder_text="Save location...")
        self.entry_output.pack(side="left", padx=10)

        btn_browse_out = ctk.CTkButton(frame_out, text="Browse", width=80, command=self.browse_output_path)
        btn_browse_out.pack(side="left", padx=10)
        self.btn_hide = ctk.CTkButton(self.tab_hide, text="ENCODE & HIDE MESSAGE", height=40, fg_color="green", hover_color="darkgreen", command=self.process_hide)
        self.btn_hide.pack(fill="x", padx=50, pady=20)
    
    def browse_cover_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if filename:
            self.hide_cover_path.set(filename)

    def browse_output_path(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if filename:
            self.hide_output_path.set(filename)




    def setup_reveal_ui(self):

        frame_stego = ctk.CTkFrame(self.tab_reveal)
        frame_stego.pack(fill="x", padx=10, pady=20)
        ctk.CTkLabel(frame_stego, text="Stego Image:").pack(side="left", padx=10)

        self.entry_stego = ctk.CTkEntry(frame_stego, textvariable=self.reveal_stego_path, width=350, placeholder_text="Select image with hidden text...")
        self.entry_stego.pack(side="left", padx=10)
        
        btn_browse_stego = ctk.CTkButton(frame_stego, text="Browse", width=80, command=self.browse_stego_image)
        btn_browse_stego.pack(side="left", padx=10)

        self.btn_reveal = ctk.CTkButton(self.tab_reveal, text="DECODE & REVEAL", height=40, fg_color="#D35B58", hover_color="#C74B48", command=self.process_reveal)
        self.btn_reveal.pack(fill="x", padx=50, pady=10)

        lbl_result = ctk.CTkLabel(self.tab_reveal, text="Decoded Message Content:", anchor="w")
        lbl_result.pack(fill="x", padx=20, pady=(20, 0))

        self.txt_result = ctk.CTkTextbox(self.tab_reveal, height=200)
        self.txt_result.pack(fill="x", padx=20, pady=5)

        self.txt_result.configure(state="disabled") # Make read-only initially

    def browse_stego_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])

        if filename:
            self.reveal_stego_path.set(filename)

    def process_hide(self):
        #print("Hide button clicked.")

        # get the input 
        cover_path = self.hide_cover_path.get()
        output_path = self.hide_output_path.get()
        message = self.txt_message.get("1.0", "end-1c") # Get text excluding auto-newline


        if not cover_path:
            messagebox.showwarning("Missing Input", "Please select a Cover Image.")
            return
        if not output_path:
            messagebox.showwarning("Missing Input", "Please select where to save the Output.")
            return
        if not message.strip():
            messagebox.showwarning("Missing Input", "Please enter a secret message.")
            return

        # my fav try and except powers
        try:
            self.processor.hide_message(cover_path, message, output_path)
            messagebox.showinfo("Success", f"Message hidden successfully!\nSaved to: {output_path}")
            
            #clear the message after success
            self.txt_message.delete("1.0", "end")
            
        except ValueError as e:
            #if message is too large
            messagebox.showerror("Capacity Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def process_reveal(self):

        #print("Reveal button clicked!!")

        stego_path = self.reveal_stego_path.get()

        if not stego_path:
            messagebox.showwarning("Missing Input", "Please select a Stego Image to decode. please or else!!")
            return

        try:
            self.txt_result.configure(state="normal") # we wanna write in it
            self.txt_result.delete("1.0", "end")
            
            revealed_msg = self.processor.reveal_message(stego_path)

            if revealed_msg:
                self.txt_result.insert("1.0", revealed_msg)
                messagebox.showinfo("Success", "Hidden message found, Here you go Champ!")
            else:
                self.txt_result.insert("1.0", "[No hidden message found or data is corrupted]")
                messagebox.showwarning("Result", "No hidden message detected.")
            
            # Disable text box again so user can't type in it haha
            self.txt_result.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


# Interactive menu.
if __name__ == '__main__':
    # steganographer = DCTSteganography()
    # try:
    #     script_dir = os.path.dirname(os.path.realpath(__file__))
    #     MESSAGE_FILENAME = "message.txt"
    #     MESSAGE_FILE_PATH = os.path.join(script_dir, MESSAGE_FILENAME)
    # except NameError:
    #     MESSAGE_FILENAME = "message.txt"
    #     MESSAGE_FILE_PATH = MESSAGE_FILENAME

    # while True:
    #     print("\n--- DCT Steganography Menu ---")
    #     print(f"1. Hide message from '{MESSAGE_FILENAME}' (Encode)")
    #     print("2. Reveal a message (Decode)")
    #     print("3. Exit")
    #     choice = input("Enter your choice (1, 2, or 3): ")

    #     if choice == '1':
    #         try:
    #             cover_path = input("Enter path to the cover image (e.g., cover.png): ")
    #             if not os.path.exists(cover_path): print("ERROR: Cover image not found."); continue
                
    #             print(f"Attempting to load message from '{MESSAGE_FILE_PATH}'...")
    #             if not os.path.exists(MESSAGE_FILE_PATH): print(f"ERROR: Message file '{MESSAGE_FILENAME}' not found."); continue

    #             with open(MESSAGE_FILE_PATH, 'r', encoding='utf-8') as f: message = f.read()
    #             if not message: print("ERROR: The message file is empty."); continue

    #             stego_path = input("Enter the output path for the stego-image (e.g., stego.png): ")
    #             steganographer.hide_message(cover_path, message, stego_path)
    #         except (FileNotFoundError, ValueError) as e: print(f"\nERROR: {e}")
    #         except Exception as e: print(f"\nAn unexpected error occurred: {e}")
        
    #     elif choice == '2':
    #         try:
    #             stego_path = input("Enter path to the stego-image (e.g., stego.png): ")
    #             if not os.path.exists(stego_path): print("ERROR: Stego-image not found."); continue
    #             revealed_message = steganographer.reveal_message(stego_path)
    #             if revealed_message is not None:
    #                 print("\n---------------------------------")
    #                 print("SUCCESS: Revealed message found!")
    #                 print("---------------------------------")
    #                 print(revealed_message)
    #                 print("---------------------------------")
    #             else:
    #                 print("\nINFO: No hidden message found or the data is corrupt.")
    #         except FileNotFoundError as e: print(f"\nERROR: {e}")
    #         except Exception as e: print(f"\nAn unexpected error occurred: {e}")
            
    #     elif choice == '3':
    #         print("Exiting program. Goodbye!"); break
    #     else:
    #         print("Invalid choice. Please enter 1, 2, or 3.")

    app = StegoAPP()
    app.mainloop()