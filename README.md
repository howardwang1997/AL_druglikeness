# activechem
# requirments
conda install
python=3.6
cudatoolkit
cudnn
numpy
pandas
rdkit=2020.03.2
scipy
torch=1.5.0
torchvision=0.6.0
scikit-learn=0.22.2
****
pip install
alipy=1.2.1
descriptastorus=2.2.0.3

# train or test
change directory to ./activechem/  
run train.py with python following with parameters: 
# parameters with * are compulsory
*-d or --dataset <dataset>: <dataset> in .sdf, .txt, .smi, .csv or dumped with joblib
*-l or --labels <labels>: <labels> in .txt or dumped with joblib
*-s or --save <saving_path>
*-f or --fold <number_of_fold>: integer larger than 1, usually smaller than 10 / -r or --ratio <test_ratio>: float smaller than 1, usually larger than 0.1
-R or --regression: enter if training a regressor model otherwise classifier
-m or --model <pretrained_model>: <pretrained_model> is pytorch model

example:
python train.py -d molecules.sdf -l labels.txt -s ./admet/ -r 0.3

