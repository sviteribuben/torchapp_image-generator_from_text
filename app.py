import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from auth_token import AuthToken
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app

app = tk.Tk()
app.geometry("532x632")
app.title("Sviter&Buben")
ctk.set_appearance_mode("dark")
# input area
promt = ctk.CTkEntry(height=40,
                     width=512,
                     text_font=("Arial", 16),
                     text_color="black",
                     fg_color="white")
promt.place(x=10, y=10)
# placeholder for image
lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid,
                                               revision="fp16",
                                               torch_dtype=torch.float16,
                                               use_auth_token=True)
pipe.to(device)

def generate():
    '''тут необходимо регистрироваться на хаггинг фейс и запускать
    в терминале кучу команд.
    pip install transformers
    pip install huggingface_hub
     huggingface-cli login
     git config --global credential.helper store
    потом пойдет подгрузка необходимых файлов для старта работы логики
'''
    with autocast(device):
        image = pipe(promt.get(), guidance_scale=8.5)["sample"][0]
    image.save("generateimage.png")
    img = ImageTk.PhotoImage(image)

    lmain.configure(image=img)

# button
trigger = ctk.CTkButton(height=40,
                        width=120,
                        text_font=("Impact", 16),
                        text_color="white",
                        fg_color="red",
                        hover_color="black",
                        command=generate
                        )
trigger.configure(text='[_EBASH_]')
trigger.place(x=206, y=60)



app.mainloop()