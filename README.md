# ST_SHAP

This is a PyTorch implementation of the ***ST-SHAP:A novel explainable attention network for EEG-based emotion recognition***.
# ST-SHAP： Architecture
![](https://github.com/llljinjinjin/ST_SHAP_code/blob/main/ST_SHAP.png)
In the figure above (a), the original EEG signal is preprocessed, then topologically interpolated according to the electrode distribution location, and emotion recognition is performed by (b) Swin Transformer. Finally, the model decision is explained by (c) SHAP method.

This study is the first to use SwinTransformer as an emotional EEG recognition model to apply concepts from the field of computer vision (CV) to the field of EEG based emotion recognition. The aim is to deeply explore the local and global dependencies in the spatial domain of affective EEG, and to interpret the lead information.

## Resources
- [ ] SEED:[LINK](https://bcmi.sjtu.edu.cn/~seed/index.html)
- [ ] DREAMER:[LINK](https://ieeexplore.ieee.org/abstract/document/7887697)

## Instructions
### Install the dependencies
It is recommended to create a virtual environment with python version 3.7 and activate it before running the following:
```
pip install -r requirements.txt
```

### Obtain the raw dataset
- [ ] Download the raw dataset from the [resources](#resources) above, and save them to the `data_input` folder.  Please download the SEED/DREAMER data in mat file format.
- [ ] Organize the raw data file into the following file structure:
```
 DatasetDir/dataset
                -/1_20131027.mat
                -/1_20131030.mat
                -/1_20131107.mat
                -/2_20140404.mat
                /...
          -/label
                -/label.mat
                .
          -/DREAMER.mat          
```
Specifically, for the SEED dataset, it contains the DATASET file and the label file, which hold the data and label, respectively. For the DREAMER file, it contains the data and labels for that dataset.

### Pre-process the raw dataset
To pre-trained SEED, run:
```
process_new.py
```
To pre-trained DREAMER, run:
```
process_dreamer.py
```
Please create folders `input_4` and `dreamer_all` in the `data_input` directory to store the processed data.  

### Model training
For SEED, run:
```
main_dreamer_swin_gamma.py
```
For DREAMER, run;
```
main_seed_swin_gamma.py
```
Please save the model files of the corresponding data set respectively in the `seed` and `dreamer` files under the `model` directory, and please unify the format of ***pth***.

### Model explaining
For SEED, please load the above model file stored in the `model/seed` folder and the preprocessed data stored in `data_input/input_4` respectively, and run:
```
shap_explain_mean.py
```
For DREAMER, please load the above model file saved in the `model/dreamer` folder and the preprocessed data saved in `data_input/dreamer_all` and run:
```
shap_explain.py
```

## Results
The classification results for our method and other competing methods are as follows:
![](https://github.com/llljinjinjin/ST_SHAP_code/blob/main/result.png)

## Cite:
If you find this architecture or toolbox useful then please cite this paper:
ST-SHAP:A novel explainable attention network for EEG-based emotion recognition

## References:
ST-SHAP:A novel explainable attention network for EEG-based emotion recognition



