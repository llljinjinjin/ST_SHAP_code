# ST_SHAP

This is a PyTorch implementation of the ***ST-SHAP:A novel explainable attention network for EEG-based emotion recognition***.
# ST-SHAP： Architecture
![](https://github.com/llljinjinjin/ST_SHAP_code/blob/main/ST_SHAP.png)
A schematic illustration of our proposed ST-SHAP method. (a) 3D spatial-temporal feature generation (b) Swin Transformer for emotional EEG classification (c) Explainability analysis based on SHAP.

This study is the first to use SwinTransformer as an emotional EEG recognition model to apply concepts from the field of computer vision (CV) to the field of EEG based emotion recognition. The aim is to deeply explore the local and global dependencies in the spatial domain of affective EEG, and to interpret the lead information.

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
Specifically, for the SEED dataset, it contains the DATASET file and the label file, which hold the data and label, respectively.

### Pre-process the raw dataset
To pre-trained SEED, run:
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

### Model explaining
For SEED, please load the above model file stored in the `model/seed` folder and the preprocessed data stored in `data_input/input_4` respectively, and run:
```
shap_explain_mean.py
```

## Results
The classification results for our method and other competing methods are as follows:
### SEED
<div align="center">
 
| Dataset | Test mode | GSCCA |BiDANN |BiHDM|R2G-STNN|DGCNN|RGNN| SST-EmotionNet|MFBPST-3D-DRLF|ST-SHAP|
| ---------- | -----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| SEED  | 40% testing, 60% training   | 82.96±9.95  |92.38±7.04|93.12±6.06|93.38±5.96|90.40±8.49|94.24±5.95|96.02±2.17|96.67±2.8|97.18±2.7|
</div>

### DREAMER
<div align="center">
 
| Methodology |  Test mode | Conti-CNN |SVM|DT |GECNN|BDAE|THR|DGCNN|RF|Deep-forest|3DFR-DFCN|Bi-AAN|GLFANet|ST-SHAP|
| ---------- |-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Arousal(%) | ten cross-validation|82.48 |84|84.48|84.54|86.60|88.20|88.79|89.58|90.41|91.30|92.95|94.82|96.06|
| Valance(%) | ten cross-validation| 81.72 |84.63|84.77|86.23|88.60|90.43|88.87|89.42|89.03|93.15|92.68|94.57|96.07|
</div>


## Cite:
If you find this architecture or toolbox useful then please cite this paper:
ST-SHAP:A novel explainable attention network for EEG-based emotion recognition

## Acknowledgment
We thank Liu Z et al and Lundberg S M et al for their wonderful works.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.[[LINK]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing
systems, 30, 2017.[[LINK]](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)





