# ydb-hnsw
A vector db implementation in Nim with YottaDB as database backend

# Python3 install

# Create a Virtual Environment
python3 -m venv hnsw_env
source hnsw_env/bin/activate

# Install torch CPU-Only version:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Test
In the ./python directory check torch installation with
 python3 verify.py

 # Install transformers
 pip3 install transformers
 pip3 install sentence_transformers
 pip3 install sentencepiece