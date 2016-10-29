s="deb http://cran.ism.ac.jp/bin/linux/ubuntu vivid/"
grep -q "$s" /etc/apt/sources.list || sudo sh -c "echo $s >> /etc/apt/sources.list"
gpg --keyserver keyserver.ubuntu.com --recv-key 51716619E084DAB9
gpg -a --export 51716619E084DAB9 | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y r-base=3.2.2-1vivid0
sudo apt-get install -y unzip

sudo apt-get install -y python-pip python-dev
pip install numpy scipy pandas scikit-learn
