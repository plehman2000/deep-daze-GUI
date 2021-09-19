import numpy as np, pandas as pd, streamlit as st
from pprint import pprint
from PIL import Image
from deep_daze import Imagine
import os
import streamlit as st
import subprocess
import os
import torch
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys
from time import sleep

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

# =============================================================================
# 
# with st_stdout("code"):
#     print("Prints as st.code()")
# =============================================================================

# =============================================================================
# with st_stdout("info"):
#     print("Prints as st.info()")
# 
# with st_stdout("markdown"):
#     print("Prints as st.markdown()")
# 
# with st_stdout("success"), st_stderr("error"):
#     print("You can print regular success messages")
#     print("And you can redirect errors as well at the same time", file=sys.stderr)
# 
# 
# =============================================================================



def runModel(current_args):

    imagine_modular = Imagine(text=current_args["text"],
                              image_width=current_args["image_width"],
                              num_layers=current_args["num_layers"],
                              batch_size=current_args["batch_size"],  # default might be 4?
                              epochs=current_args["epochs"],
                              open_folder=current_args["open_folder"],
                              save_progress=current_args["save_progress"],
                              save_video=current_args["save_video"],
                              gradient_accumulate_every=current_args["gradient_accumulate_every"],
                              save_every=current_args["save_every"],
                              save_gif=current_args["save_gif"],
                              center_bias=current_args["center_bias"],
                              jit=current_args["jit"])  # default is 16
    
    torch.cuda.empty_cache()
    imagine_modular()
    #print("Model has Finished!")
# =============================================================================
#     
#     process = subprocess.Popen(args=["python",  os.path.join(
#     path, 'model_run.py'), f' {input_text}'], shell=True, close_fds=True)
# =============================================================================
    #add args for model parameters

def deleteImages():
    image_paths  = r'C:\Coding\Anaconda\envs\GANSPACE\myScripts'
    files = os.listdir(image_paths)
    files = os.listdir(image_paths)

    paths = [os.path.join(image_paths, basename) for basename in files]
    filtered_paths = []
    for path in paths:
        #print(path + "   " + path[-3:] + '\n\n')
        if(path[-3:] == "jpg") or (path[-3:] == "mp4"):
            os.remove(path)
    
    
def updateImage():
    image_paths  = r'C:\Coding\Anaconda\envs\GANSPACE\myScripts'
    with st.spinner(text='In progress...'):
        files = os.listdir(image_paths)
    
        paths = [os.path.join(image_paths, basename) for basename in files]
        filtered_paths = []
        for path in paths:
            #print(path + "   " + path[-3:] + '\n\n')
            if(path[-3:] == "jpg"):
                filtered_paths.append(path)
        #print("Paths filtered:", filtered_paths)
        if(filtered_paths):
            latest_file = max(filtered_paths, key=os.path.getctime)
            if(latest_file):
                name = latest_file[len(image_paths)+1:-4].replace('_', ' ')
                if name[-6] == "0":
                    name = name[:-7]
                st.write(f"'{name}'")
                print("LATEST FILE:  ", latest_file)
                im = Image.open(latest_file)
                st.image(image=im)


path = r'C:\Coding\Anaconda\envs\GANSPACE\myScripts'
st.title('Deep Daze Image Generator')
st.session_state.input = st.text_input("Enter a prompt for the model to base an image off of:", "")
current_args = {
    #takes bout 1 min per epoch
    "lr":1e-3,
    "text": None,
    "image_width": 256,
    "num_layers": 8,
    "batch_size": 4,#default is 4
    "epochs": 1,
    "gradient_accumulate_every": 1,#default is 4
    "iterations": 100,
    "save_every":100,
    "open_folder":False,
    "save_progress": True,
    "save_gif": False,
    "center_bias":False,
    "save_video": False,
    "jit": False}
    
#i want image width options, epochs, learning rate, save video
# =============================================================================
placeholder = st.expander(label='Parameters (Click here to expand/collapse)',expanded=True)
title_ratio = [2, 2, 1]
slider_ratio = [0.5, 1]

placeholder.info("Note: A model running with default parameters takes approximately one minute per epoch")


col1, col2, col3 = placeholder.columns(title_ratio)
col2.write("Learning Rate")
colRate, colslider = placeholder.columns(slider_ratio)

current_args["lr"] =  10 ** (colslider.slider(label="",min_value=-8, max_value=-1, value=-3, step=1))
colRate.success(current_args["lr"])

col1, col2, col3 = placeholder.columns(title_ratio)
col2.write("Image Width")
colImageSize, colslider2 = placeholder.columns(slider_ratio)

current_args["image_width"] = (colslider2.slider(label="",min_value=128, max_value=1024, value=256, step=128))
colImageSize.success(str(current_args["image_width"]) + " x " + str(current_args["image_width"]) + " pixels")

col1, col2, col3 = placeholder.columns(title_ratio)
col2.write("Number of Epochs")
colEpochs, colslider3 = placeholder.columns(slider_ratio)
current_args["epochs"] = (colslider3.slider(label="",min_value=1, max_value=10, value=1, step=1))
colEpochs.success(str(current_args["epochs"]) + " Epochs")


if placeholder.checkbox("Save Video"):
    current_args["save_video"] = True


if placeholder.checkbox("Open Image Folder"):
    current_args["open_folder"] = True
    
    
col1, col2 = st.columns([1,1])


if col1.button('Generate Image'):
    #st.button("Check for new output", on_click = check_dir())  
    empty = st.empty()
    empty.info(f'Running model using input: "{st.session_state.input}"')
    current_args["text"] = st.session_state.input
    runModel(current_args)  
    #sleep(3)
    empty.empty()
    st.info("Model has Finished!")
    updateImage()

if col2.button('Delete Output Files'):
    deleteImages()






