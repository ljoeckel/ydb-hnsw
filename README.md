# ydb-hnsw
A vector db implementation in Nim with YottaDB as database backend.
It is implemented by using a HNSW index.
A Hierarchical Navigable Small World (HNSW) Index is a highly efficient, graph-based algorithm for the approximate search for the nearest neighbors (Approximate Nearest Neighbor, ANN) in high-dimensional vector data. It is mainly used in modern vector databases and AI applications

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