'''
Paulina Pérez Garcés

main.py is the main file of the project. It imports all the necessary modules 
and runs the algorithms. It also creates a virtual environment and installs all
the necessary packages.
'''

if __name__ == "__main__":

    #Basic imports for creating the virtual environment
    from subprocess import run

    osInfo = input(Please indicate your OS: Linux (L), MacOS (M), Windows (W) \n')

    while osInfo not in ['L', 'M', 'W']:
        osInfo = input('Please indicate your OS: Linux (L), MacOS (M), Windows (W) \n')

    if createvenv=='M' or createvenv=='L':
        #Creation of the virtual environment
        run(["python", "-m", "venv", "venvIA"])
        #Installation of all the necessary packages
        run(["venvIA/bin/python", "-m", "pip", "install", "--upgrade", "pip"])
        run(["venvIA/bin/pip", "install", "-r", "./src/requirements.txt"])

    if createvenv=='W':
        #Creation of the virtual environment
        run(["python", "-m", "venv", "venvIA"])
        #Installation of all the necessary packages
        run(["venvIA/Scripts/python.exe", "-m", "pip", "install", "--upgrade", "pip"])
        run(["venvIA/Scripts/pip.exe", "install", "-r", "./src/requirements.txt"])

    #Runing the project files using the virtual environment
    run(["venvIA/bin/python", "src/ppl.py"])