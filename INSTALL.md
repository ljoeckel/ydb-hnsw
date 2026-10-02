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

# Install ydb-hnsw
git clone ..... xxxxx yyyyc

### Test installation
Logout and login again
ydb -v
nim -v
