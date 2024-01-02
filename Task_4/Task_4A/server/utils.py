ALLOWED_EXTENSIONS = {"h5", "pt", "keras", "pth", "zip", "tf"}


def is_allowed_file_ext(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
