# Metal3DRNA
Metal3DRNA is a method for the prediction of metal ion-binding sites in RNA structure.
Authors are YanPeng Zhao, Weikang Gong, Jingjing Wang, Yang Liu and Chunhua Li.

The pdb, model and result directories contain the four tested structures, the trained models and generated results, respectively. The depended python packages are listed in requirements.txt. The package versions should be followed by users in their environments to achieve the supposed performance.

***

## How to run

The program should be run in Python 3.6 using Tensorflow 2.0 backends. 

The first step: Put the PDB files to be predicted into the PDB folder

The second step: Use the following bash command to run Metal3DRNA.
```
    python independent_test.py -s structure -m model(optional)
```
The parameter of -s should be the file name of the PDB structure you want to predict.

The parameter of -m could be mg, na and k which mean the ion types you want to predict. 

Then, Metal3DRNA will perform predict process, which will take a few minutes.

The results will be given in the result folder, and be named as the model of the ion type appended by the time when the prediction begins.

***

## Description of result files

After the prediction is completed, the user will see the prediction results in Excel file under the result folder.

The first column of Excel file is the 3D coordinates of the grid points (RNA structure is projected on a 3D grid), the second column is the file name, the third column is the probability of the gird point being predictd as a negative sample, and the fourth column is the probability of the grid point being predicted as a positive sample. 

***

## Help

For any questions, feel free to contact the author by zhaoyp@emails.bjut.edu.cn or chunhuali@bjut.edu.cn.


