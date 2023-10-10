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
  pip install -r requirements.txt
### Obtain the raw dataset
Download the raw dataset from the [resources](#Resources) above, and save them to the same folder.  Please download the SEED/DREAMER data in mat file format.
