import os
import streamlit as st
from datetime import datetime
import subprocess
import cv2

st.title("Task 4A: Model Injection")
st.header("I'm telling you, don't inject weird stuff, or I might anger you")

file = st.file_uploader("Upload your model here")
# write file to disk
if file is not None:
    addr = f"model_{datetime.now().timestamp()}.h5"
    with open(addr, "wb") as f:
        f.write(file.read())

    st.write(f"Model saved at {addr}, use this in your inference and load code.")

st.write("Upload your model here")

text_load = st.text_area("Model Load Code (Should return a model via 'return model').")
text_inference = st.text_area("Model Inference Code (In: 'imagepath', Out: 'classmap[classid]', 'classmap' is global variable)")

addr = ""
if text_load is not None and text_inference is not None:
    addr = f"temp_task_4a_{datetime.now().timestamp()}.py"
    st.write(f"Saving code to {addr}")
    with open(addr, "w") as f:
        with open("task_4a_online_template.py", "r") as template:
            # add 4 space indent to each line of text_load
            text_load = "\n".join(["    " + line for line in text_load.split("\n")])
            # add 4 space indent to each line of text_inference
            text_inference = "\n".join(["    " + line for line in text_inference.split("\n")])
            templ = template.read()
            templ.replace("    # -->LOL<<--[[{{LOAD_MODEL}}]]-->>LOL<<--", text_load)
            templ.replace("    # -->LOL<<--[[{{CLASSIFY_EVENT}}]]-->>LOL<<--", text_inference)
            templ.replace("__{{REPLACE THIS}}__", addr)
            f.write(templ)

    st.write("Saved.")

# run the file while showing output
if addr != "":
    st.write("Running your code...")
    st.code(subprocess.run(f"conda activate GG_2907; python {addr}", capture_output=True, shell=True).stdout.decode("utf-8"))
    st.write("Done.")

# show image
im = cv2.imread(f"arena_with_labels{addr}.jpg")
st.image(im, caption="Arena with labels", use_column_width=True)
