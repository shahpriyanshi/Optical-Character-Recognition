# Handwritten-text-recognition

The work flow of the project:  
1: the input image is fed into the CNN layers to extract features. The output is a feature map. 
2: Feature map is passed through the implementation of Long Short-Term Memory (LSTM) (the RNN is able to propagate information over longer distances and provide more robust features to training)
3: RNN output matrix- the Connectionist Temporal Classification (CTC) calculates loss values and also decodes into the final text.


Project Structure:

  1: data/:contains iam dataset .hdf5 file (Download the iam.hdf5 file from my google drive link (https://drive.google.com/drive/folders/1vmr1QjnIK-LfInlGbgACOYHjcYnWs7uB?usp=sharing) and place in the data folder if you don't want to create one)
  2: raw/: Contains iam dataset (Download the iam lines dataset and place it in a "lines/" folder in this folder.
  3: src/:
      (i): data/: contains python files for reading, preprocessing, generating batches and evaluating the iam dataset
      (ii): htr_weights3.h5 : Weight file of the project ((Download the weights file from my google drive link (https://drive.google.com/drive/folders/1vmr1QjnIK-LfInlGbgACOYHjcYnWs7uB?usp=sharing) and place in the src folder if you don't want to create one)
      (iii): evaulate.txt: results after testing the trained model on test iamges
      (iv): main.py: file for generating .hdf5 file from the iam dataset raw:/ folder containing iam lines dataset
      
      
Steps for running:

  1: .hdf5 file: run main.py file to generate .hdf5 file using dataset from raw folder
        (py -3.6 main.py --source=iam --transform) new "data" folder containing .hdf5 file will automatically be created on running main.py
  2: Training the model and testing the model:
     run Handwritten-Text-Recognition.ipynb
    
Results:
  --Total test images:    1425
  --Character Error Rate: 7.044 %
  --Word Error Rate:      21.892 %

(Please follow the structure for successfully running each files)
