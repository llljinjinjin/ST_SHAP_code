The SEED dataset's raw feature map data and SHAP map data are divided by **subjects** and stored separately. 

In each subject folder, we subdivided them according to the **categories** and stored them separately with the `category as the subfile name`.


The data in each sub-file has the shape of **(2,32,32)**, which is *stacked by the raw feature map array and SHAP feature map array* of the category.
