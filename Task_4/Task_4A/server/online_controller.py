import os
from flask import (
    Flask,
    flash,
    render_template,
    request,
    redirect,
    send_from_directory,
    url_for,
)
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import subprocess
import cv2
import dotenv

from utils import is_allowed_file_ext

dotenv.load_dotenv(".env")

UPLOAD_FOLDER = "temp_models_quick"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.secret_key = os.getenv("SECRET_KEY")

VERSION = "0.2.0-b"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("You really need to upload a file for this to work...")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "" or file.filename is None:
            flash("You really need to upload a file for this to work...")
            return redirect(request.url)

        if file:
            if not is_allowed_file_ext(file.filename):
                flash("File type not allowed")
                return redirect(request.url)
            else:
                # create new filename with timestamp
                nowtime = str(datetime.now().timestamp()).replace(".", "_")
                # log IP address, time, and user agent
                with open("log.txt", "a") as f:
                    f.write(
                        f"{nowtime} - {request.remote_addr} - {request.user_agent} | OGNAM: {file.filename}\n"
                    )

                extension = file.filename.rsplit(".", 1)[1].lower()
                filename_timestamped = f"model_{nowtime}.{extension}"

                # save the file to new location
                filename = secure_filename(filename_timestamped)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

                # check if user wants to unzip
                unzip = request.form.get("unzip")

                if unzip:
                    if extension == "zip":
                        folder_out = filename.rsplit(".", 1)[0]

                        # TODO: thread this
                        subprocess.check_output(
                            f"cd temp_models_quick && unzip {filename} -d {folder_out}",
                            shell=True,
                        )

                        # now set filename to the unzipped folder, so it is shown correctly in the next page
                        filename = filename.split(".")[0]
                        return redirect(url_for("upload_code", filename=filename))

                    else:
                        flash(
                            "Cannot unzip this file type. I told you it's only for zip files..."
                        )
                        # delete file
                        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                        return redirect(request.url)
                else:
                    return redirect(url_for("upload_code", filename=filename))
        else:
            flash("This is a very strange error, please try again.")
            return redirect(request.url)

    return render_template("upload.html", version=VERSION)


@app.route("/upload-code/<filename>", methods=["GET", "POST"])
def upload_code(filename):
    if request.method == "POST":
        code_load = request.form.get("code_load")
        code_classify = request.form.get("code_classify")

        if (
            code_load == ""
            or code_classify == ""
            or code_load is None
            or code_classify is None
        ):
            flash("Please enter code for both loading and classifying")
            return redirect(request.url)
        else:
            # create file f"temp_code_quick/task_4a_{filename}.py"
            with open(f"{UPLOAD_FOLDER}/task_4a_{filename}.py", "w") as f:
                with open("task_4a_online_template.py", "r") as template:
                    # add 4 space indent to each line of text_load
                    code_load = "\n".join(
                        ["    " + line for line in code_load.split("\n")]
                    )
                    # add 4 space indent to each line of text_inference
                    code_classify = "\n".join(
                        ["    " + line for line in code_classify.split("\n")]
                    )
                    templ = template.read()
                    templ = templ.replace(
                        "    # -->LOL<<--[[{{LOAD_MODEL}}]]-->>LOL<<--", code_load
                    )
                    templ = templ.replace(
                        "    # -->LOL<<--[[{{CLASSIFY_EVENT}}]]-->>LOL<<--",
                        code_classify,
                    )
                    templ = templ.replace("__{{REPLACE THIS}}__", filename)
                    f.write(templ)

            return redirect(url_for("run_code", filename=filename))

    return render_template("upload_2.html", filename=filename, version=VERSION)


@app.route("/run-code/<filename>", methods=["GET"])
def run_code(filename):
    print("running code...")
    timestart = datetime.now()
    try:
        # TODO: thread this
        pr = subprocess.check_output(
            f"conda activate GG_2907 && cd temp_models_quick && python task_4a_{filename}.py",
            shell=True,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        timeend = datetime.now()
        output = pr
        Aimg = f"A{filename}.png"
        Bimg = f"B{filename}.png"
        Cimg = f"C{filename}.png"
        Dimg = f"D{filename}.png"
        Eimg = f"E{filename}.png"
        filename = f"arena_with_labels{filename}.jpg"
        success = True
    except subprocess.CalledProcessError as e:
        output = str(e) + "\nreturncode: " + str(e.returncode) + "\n" + str(e.output)
        Aimg = Bimg = Cimg = Dimg = Eimg = filename = f"crazy.png"
        success = False
        timeend = datetime.now()

    print("run over, rendering template...")

    timetaken_pretty = str(timeend - timestart)

    return render_template(
        "run_code.html",
        filename=filename,
        imgs=[Aimg, Bimg, Cimg, Dimg, Eimg],
        code_out=output,
        success=success,
        version=VERSION,
        timetaken=timetaken_pretty,
    )


@app.route("/show-image/<filename>")
def show_image(filename):
    return send_from_directory("temp_models_quick", filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=False)
