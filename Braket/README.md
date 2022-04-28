# Running PolarQaoa in Amazon Braket
This notebooks in the directory show how to run Quantum Channel Decoders with real or simulated quantom computers on Amazon Braket. This code is based on our paper "Quantum Channel Decoding" as submitted to the [QEC-22 conference](https://qce.quantum.ieee.org/2022/)

## Running locally
### Create and initialize a virtual environment
Make sure you have python 3.8 or later installed. Then run the following commands in a termonal window:
```
python3 -m venv ve
source ve/bin/activate
pip install --upgrade pip setuptools wheel numpy pyyaml matplotlib scipy notebook
pip install amazon-braket-sdk
```

### To run the notebook files
In a terminal window, go to directory containing this code and run:
```
jupyter notebook
```
or if running on a remote server:
```
jupyter notebook --ip <IP_ADDRESS>
```

## Running on Amazon Braket
### Create Notebook
In your Amazon Braket dashboard, go to ``Notebooks`` and create a new notebook. Once it becomes available click on the link to open the notebook in a new browser window.

### Upload files:
In the Jupyter notebook, upload the following files:
- QuantumPolar.ipynb
- QuantumHamming.ipynb
- quantumpolardecoder.py
- quantumhammingdecoder.py
- polarsc.py

### Once the upload is complete, click on the ``QuantumPolar.ipynb`` or ``QuantumHamming.ipynb`` file.
