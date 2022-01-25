from __future__ import division
import os
import shutil

def setup_folder(folder, parameter_file_name, extra_files=[]):
    "Makes simulation directory and copies files for later reference."
    if os.path.exists(folder):
        print("Deleting simulation folder.")
        shutil.rmtree(folder)
    os.makedirs(folder)
    shutil.copy(parameter_file_name, folder)
    for file in extra_files:
        shutil.copy(file, folder)
    os.chdir(folder)
    print("Setup directory.")