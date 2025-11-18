1- create virtual environment.
	py -3.10 -m venv <name>

2- activate the venv

3- nav to the project folder

4- python -m pip install --upgrade pip

5- python -m pip install "torch==2.1.2+cpu" -f https://download.pytorch.org/whl/torch_stable.html

6- python -m pip install "onnx==1.15.0" "onnxruntime==1.16.3" "protobuf==3.20.3" "numpy<1.24"

7- python -m pip install ./ml-agents-env

8- python -m pip install ./ml-agents

9- python run_all.py