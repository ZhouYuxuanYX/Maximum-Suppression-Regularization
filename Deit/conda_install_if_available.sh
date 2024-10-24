while read requirement
do conda install --yes $requirement || pip3 install $requirement
done < requirements.txt