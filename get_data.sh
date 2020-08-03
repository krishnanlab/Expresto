home_dir=$(dirname $0)
cd $home_dir

if [ -d data ]; then
	echo cleaning up old data
	rm -rf ./data/*
else
	mkdir data
fi

wget https://zenodo.org/record/3971092/files/2020-08-03-data.tar.gz
tar -xvzf 2020-08-03-data.tar.gz -C data --strip-components=1
rm -f 2020-08-03-data.tar.gz
