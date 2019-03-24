import os

path = '../experiment/test/results-B100'

for png in os.listdir(path):
    file_path = os.path.join(path, png)
    if 'x1' in png:
        new_file_path = png.split('x')[0]
        os.rename(file_path, os.path.join(path, new_file_path+".png"))
