# Install YottaDB
### Installing prerequisite packages
apt-get install -y --no-install-recommends file binutils libelf-dev libicu-dev nano wget
### Get YottaDB Installer
mkdir /tmp/tmp
cd /tmp/tmp
wget https://download.yottadb.com/ydbinstall.sh
chmod +x ydbinstall.sh
### Run installer
sudo ./ydbinstall.sh --utf8 --verbose
### Configure .bashrc
Add to .bashrc
#---- Init YottaDB environment
. /usr/local/lib/yottadb/r206/ydb_env_set

# Install Nim
Run
apt install gcc
curl https://nim-lang.org/choosenim/init.sh -sSf | sh
### Test installation
Logout and login again
ydb -v
export PATH=/home/<name>/.nimble/bin:$PATH

# Install llama.cpp (embedding backend)
Embeddings are computed in-process by llama.cpp from a GGUF file - no Python at
runtime.
```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build llama.cpp/build -j --target llama-embedding
```
For the GPU add `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<cc>` (61 for a
GTX 1060) to the configure step and apply `bash/llama_cpp_cuda_cmake.patch`
first on CMake 4.x. The binary offloads automatically when llama.cpp reports a
GPU; `HNSW_EMBED_GPU_LAYERS=0` pins the CPU.
Add to .bashrc (config.nims reads the first, llama_embed the second):
```
export LLAMA_CPP_DIR=$HOME/git/llama.cpp/build
export HNSW_GGUF=$HOME/models/gguf/paraphrase-multilingual-MiniLM-L12-v2.gguf
```
# or HNSW_GGUF_DIR=<dir> and the file is looked up as <model name>.gguf

Get the model file once (needs python3 + torch/transformers for the converter
only - see README, and apply bash/llama_cpp_xlmroberta.patch first):
```bash
python llama.cpp/convert_hf_to_gguf.py \
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --outtype q8_0 --outfile "$HNSW_GGUF"
```
q8_0 is the fastest choice with a GPU; use --outtype f16 without one. Index and
queries must use the same file.

# Install ydb-hnsw
git clone ..... xxxxx yyyyc

### Test installation
Logout and login again
ydb -v
nim -v
