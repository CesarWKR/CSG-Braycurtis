# Description

This project is focused on analyzing the complexity of a dataset by calculating various similarity metrics at different levels. Specifically, it aims to:

**Dataset Complexity:** Quantify the overall complexity of the dataset using the CSG (Cumulative Gradient Score) metric, which provides a detailed measure of the dataset's structure and separability.
**Class Similarity:** Compute the similarity between different classes within the dataset to evaluate their overlap or distinctiveness.
**Sample Similarity:** Measure the similarity between individual samples, either within the same class or across the entire dataset.

To achieve these goals, the project integrates the Bray-Curtis similarity metric with the CSG framework. This integration enables precise calculations of similarity both at the class level and at the sample level, 
providing a comprehensive understanding of the relationships within the dataset. These insights are particularly useful for assessing dataset quality, identifying redundant or overlapping data, and understanding 
potential challenges for machine learning models.



The first step is to install the library using pip as follows:

```pip install new_spectral_metric``` 

Once installed, you can access the files that make up the library. The library provides functionality to compute similarity metrics at different levels:

Intra-class similarity: Use new_estimator_intra_class (available within new_spectral_metric) to calculate the similarity between samples within the same class.
Dataset-wide similarity: Use new_estimator_all_samples to compute similarity across all samples in the dataset.
Alternatively, you can use new_estimator_complete, which combines both intra-class similarity and dataset-wide similarity into a single estimator for comprehensive analysis.


```
from new_spectral_metric import new_estimator_intra_class, new_estimator_all_samples, new_estimator_complete

# Intra-class similarity 
from new_spectral_metric.new_estimator_intra_class import CumulativeGradientEstimator_Intra_class
intra_class_similarity = new_estimator_intra_class(data, class_indices)  

# All dataset similarity 
from new_spectral_metric.new_estimator_all_samples import CumulativeGradientEstimator_All_samples
all_samples_similarity = new_estimator_all_samples(data)  

# Complete similarity 
from new_spectral_metric.new_estimator_complete import CumulativeGradientEstimator_Complete
complete_similarity = new_estimator_complete(data, class_indices) 
```


This project is licensed under the MIT License. [See the LICENSE](./LICENSE) file for more details.

# Results

You can see the tests performed in the folder [CSG and similarity](./CSG%20and%20similarity/).
Inside this folder you will find the Jupiter optimized codes for each type of similarity (Intra-class, All dataset and combined).