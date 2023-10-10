# ST_SHAP
## ST-SHAP:A novel explainable attention network for EEG-based emotion recognition

This is a PyTorch implementation of the ST-SHAP architecture for emotional EEG classification.
# ST-SHAP： Architecture
![](https://github.com/llljinjinjin/ST_SHAP_code/blob/main/ST_SHAP.png)
In the figure above (a), the original EEG signal is preprocessed, then topologically interpolated according to the electrode distribution location, and emotion recognition is performed by (b) Swin Transformer. Finally, the model decision is explained by (c) SHAP method.

## Resources
* SEED:[LINK](https://bcmi.sjtu.edu.cn/~seed/index.html)
- DREAMER:[LINK](https://ieeexplore.ieee.org/abstract/document/7887697)

## Instructions
### Install the dependencies
It is recommended to create a virtual environment with python version 3.7 and activate it before running the following:
```
pip install -r requirements.txt
```

### Obtain the raw dataset
Download the raw dataset from the [resources](#resources) above, and save them to the `data_input` folder.  Please download the SEED/DREAMER data in mat file format.

### Pre-process the raw dataset
To pre-trained SEED, run:
```
process_new.py
```
Save the preprocessed data to `./data_input/input_4`.  
  
To pre-training DREAMER, run:
```
process_dreamer.py
```
Save the preprocessed data to `./data_input/dreamer_all`.

## Results
The classification results for our method and other competing methods are as follows:
![](https://github.com/llljinjinjin/ST_SHAP_code/blob/main/result.png)

## Cite:
If you find this architecture or toolbox useful then please cite this paper:
ST-SHAP:A novel explainable attention network for EEG-based emotion recognition

## References:
ST-SHAP:A novel explainable attention network for EEG-based emotion recognition



