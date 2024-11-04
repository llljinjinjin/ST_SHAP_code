## 1. Divided by subject
The DREAMER dataset's raw feature map data and SHAP map data are divided by **subjects** and stored separately. 

## 2. Further divided by category
In each subject folder, we subdivided them according to the **categories** and stored them separately in `.npy` format with the `category as the subfile name`.


The data in each sub-file has the shape of **(2,32,32)**, which is ***stacked by the raw feature array and SHAP feature array*** of the category.

## 3. Save the raw feature data of each subject
In addition, in order to facilitate the comparison of the two-dimensional raw feature data for each category, in `each subject's folder`, we specifically saved the raw feature map data for the corresponding subject in `.xlsx format`. Where **each category in the file is a sub-table**.
