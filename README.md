# ST_SHAP

This is a PyTorch implementation of the ***ST-SHAP: A hierarchical and explainable attention network for emotional EEG representation learning and decoding***.
# ST-SHAP： Architecture
![](https://github.com/llljinjinjin/ST_SHAP_code/blob/main/ST_SHAP.png)
A schematic illustration of our proposed ST-SHAP method. (a) 3D spatial-temporal feature generation (b) Swin Transformer for emotional EEG classification (c) Explainability analysis based on SHAP.

In order to simultaneously and fully capture the local characteristics and global dependencies of emotional EEG data on spatial-temporal domains, we explore the adaptation of Swin Transformer for EEG-based emotion recognition. Furthermore, the explainability of the Swin Transformer based emotional EEG decoding model is deeply investigated via the SHapley Additive exPlanations (SHAP) method and neurophysiological prior knowledge. As a whole, we propose a novel hierarchical and explainable attention network, ST-SHAP, for EEG-based emotion recognition.

## Resources
* SEED:[LINK](https://bcmi.sjtu.edu.cn/~seed/index.html)

## Instructions
### Install the dependencies
It is recommended to create a virtual environment with python version 3.7 and activate it before running the following:
```
pip install -r requirements.txt
```

### Obtain the raw dataset
* Download the raw dataset from the [resources](#resources) above, and save them to the `data_input` folder.  Please download the SEED data in mat file format.
- Organize the raw data file into the following file structure:
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
```
Specifically, for the SEED dataset, it contains the dataset file and the label file, which hold the data and label, respectively.

### Pre-process the raw dataset
To pre-process SEED, run:
```
process_new.py
```
Please create folders `input_4`  in the `data_input` directory to store the processed data.  

### Model training
For SEED, run:
```
main_seed_swin_gamma.py
```
Please save the model files of the corresponding data set respectively in the `seed` files under the `model` directory, and please unify the format of ***pth***.


## Results
The classification results for our method and other competing methods are as follows:
### SEED
<div align="center">
 
| Dataset | Test mode | GSCCA |BiDANN |BiHDM|R2G-STNN|DGCNN|RGNN| SST-EmotionNet|MFBPST-3D-DRLF|ST-SHAP|
| ---------- | -----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| SEED  | 40% testing, 60% training   | 82.96±9.95  |92.38±7.04|93.12±6.06|93.38±5.96|90.40±8.49|94.24±5.95|96.02±2.17|96.67±2.8|97.18±2.7|
</div>


## Cite:
If you find this architecture or toolbox useful then please cite this paper:
ST-SHAP: A hierarchical and explainable attention network for emotional EEG representation learning and decoding

## Acknowledgment
We thank Liu Z et al and Lundberg S M et al for their wonderful works.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing
systems, 30, 2017.[[LINK]](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)





