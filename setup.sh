# Requires Python 3.6.9
pip install -r requirements.txt

echo "Unzipping Data"
unzip new_data.zip
cd real_data_gen/
unzip vocab_phase2.zip
cd triplets/
unzip phase2.zip

echo "Downloading model weights"
cd ../../output/
wget  https://zenodo.org/record/6970062/files/epoch_29_fold_1.h5

echo "Generating all data from raw..."
cd ../
python3 ./real_data_gen/modify_data.py
