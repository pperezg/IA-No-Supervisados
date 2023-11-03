'''
Paulina Pérez Garcés

main.py is the main file of the project. It imports all the necessary modules 
and runs the algorithms. It also creates a virtual environment and installs all
the necessary packages.
'''

if __name__ == "__main__":

    #Basic imports for creating the virtual environment
    from subprocess import run

    createvenv = input('Do you want to create a virtual environment? (y/n)')

    if createvenv=='y':
        #Creation of the virtual environment
        run(["python", "-m", "venv", "venvIA"])
        #Popen(["source ", "venvIA/bin/activate"],shell=True)
        run(["venvIA/bin/python", "-m", "pip", "install", "--upgrade", "pip"])
        run(["venvIA/bin/pip", "install", "-r", "./src/requirements.txt"])

    run(["venvIA/bin/python", "src/ppl.py"])