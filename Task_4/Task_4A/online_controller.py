import os
from flask import Flask, flash, render_template, request, redirect, send_from_directory, url_for
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import subprocess
import cv2

UPLOAD_FOLDER = 'temp_models_quick/'
ALLOWED_EXTENSIONS = {'h5', 'pt', 'keras', 'pth'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists("temp_code_quick"):
    os.makedirs("temp_code_quick")

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = "e05o0504054oqp5"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            if allowed_file(file.filename):
                extension = file.filename.rsplit('.', 1)[1].lower() # type: ignore
                filename_timestamped = f"model_{str(datetime.now().timestamp()).replace('.', '_')}.{extension}"
                filename = secure_filename(filename_timestamped) # type: ignore
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                return redirect(url_for('upload_code', filename=filename))
            else:
                flash('File type not allowed')
                return redirect(request.url)
        
    return render_template("upload.html")

@app.route('/upload-code/<filename>', methods=['GET', 'POST'])
def upload_code(filename):
    if request.method == "POST":
        code_load = request.form.get("code_load")
        code_classify = request.form.get("code_classify")

        if code_load is None or code_classify is None:
            flash("Please enter both codes")
            return redirect(request.url)
        else:
            # create file f"temp_code_quick/task_4a_{filename}.py"
            with open(f"temp_models_quick/task_4a_{filename}.py", "w") as f:
                with open("task_4a_online_template.py", "r") as template:
                    # add 4 space indent to each line of text_load
                    code_load = "\n".join(["    " + line for line in code_load.split("\n")])
                    # add 4 space indent to each line of text_inference
                    code_classify = "\n".join(["    " + line for line in code_classify.split("\n")])
                    templ = template.read()
                    templ = templ.replace("    # -->LOL<<--[[{{LOAD_MODEL}}]]-->>LOL<<--", code_load)
                    templ = templ.replace("    # -->LOL<<--[[{{CLASSIFY_EVENT}}]]-->>LOL<<--", code_classify)
                    templ = templ.replace("__{{REPLACE THIS}}__", filename)
                    f.write(templ)

            return redirect(url_for('run_code', filename=filename))
        
    return render_template("upload_2.html", filename=filename)

@app.route('/run-code/<filename>', methods=['GET'])
def run_code(filename):
    print("running code...")
    try:
        pr = subprocess.check_output(f"conda activate GG_2907 && cd temp_models_quick && python task_4a_{filename}.py", shell=True).decode('utf-8')
        output = pr
        filename = f"arena_with_labels{filename}.jpg"
    except Exception as e:
        output = str(e)
        filename = "firstframe.jpg"
    print("trying to show")
    return render_template("run_code.html", filename=filename, code_out=output)

@app.route("/show-image/<filename>")
def show_image(filename):
    return send_from_directory("temp_models_quick", filename, as_attachment=True)
# st.title("Task 4A: Model Injection")
# st.header("I'm telling you, don't inject weird stuff, or I might anger you")

# file = st.file_uploader("Upload your model here")
# # write file to disk
# if file is not None:
#     addr = f"model_{datetime.now().timestamp()}.h5"
#     with open(addr, "wb") as f:
#         f.write(file.read())

#     st.write(f"Model saved at {addr}, use this in your inference and load code.")

# st.write("Upload your model here")

# text_load = st.text_area("Model Load Code (Should return a model via 'return model').")
# text_inference = st.text_area("Model Inference Code (In: 'imagepath', Out: 'classmap[classid]', 'classmap' is global variable)")

# addr = ""
# if text_load is not None and text_inference is not None:
#     addr = f"temp_task_4a_{datetime.now().timestamp()}.py"
#     st.write(f"Saving code to {addr}")
#     with open(addr, "w") as f:
#         with open("task_4a_online_template.py", "r") as template:
#             # add 4 space indent to each line of text_load
#             text_load = "\n".join(["    " + line for line in text_load.split("\n")])
#             # add 4 space indent to each line of text_inference
#             text_inference = "\n".join(["    " + line for line in text_inference.split("\n")])
#             templ = template.read()
#             templ.replace("    # -->LOL<<--[[{{LOAD_MODEL}}]]-->>LOL<<--", text_load)
#             templ.replace("    # -->LOL<<--[[{{CLASSIFY_EVENT}}]]-->>LOL<<--", text_inference)
#             templ.replace("__{{REPLACE THIS}}__", addr)
#             f.write(templ)

#     st.write("Saved.")

# # run the file while showing output
# if addr != "":
#     st.write("Running your code...")
#     st.code(subprocess.run(f"conda activate GG_2907; python {addr}", capture_output=True, shell=True).stdout.decode("utf-8"))
#     st.write("Done.")

# # show image
# im = cv2.imread(f"arena_with_labels{addr}.jpg")
# st.image(im, caption="Arena with labels", use_column_width=True)

if __name__ == "__main__":
    app.run(debug=False)
 