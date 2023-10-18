import requests
from PIL import Image 
def download_image(url):
  """Downloads an image from the given URL and saves it to the current directory with the same filename as the image."""

  response = requests.get(url)
  file_name = url.split("/")[-1]
  with open("damaged/" + file_name, "wb") as f:
    f.write(response.content)
# Retrieving the resource located at the URL 
# and storing it in the file name a.png 
with open("damaged.csv") as file:
    lines = [line.strip() for line in file.readlines()]
    for index, line in enumerate(lines):
        print(f"Getting file {index}")
        try:
            download_image(line) 
        except:
           continue

