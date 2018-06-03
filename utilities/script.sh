sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install python3
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo apt-get install -y unzip
sudo apt-get install -y python3-pip
sudo pip3 install -r requirements.txt


cd utilities
./get_word2vec_sample.sh
python3 tokenize_corpus.py --corpus=../data/word2vec_sample/text8 --vocab_size=50000 --output=text8_tokenize$

cd ..
