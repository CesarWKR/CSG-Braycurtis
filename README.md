# Description

This project is focused on analyzing the complexity of a dataset by calculating various similarity metrics at different levels. Specifically, it aims to:

**Dataset Complexity:** Quantify the overall complexity of the dataset using the CSG (Cumulative Gradient Score) metric, which provides a detailed measure of the dataset's structure and separability.
**Class Similarity:** Compute the similarity between different classes within the dataset to evaluate their overlap or distinctiveness.
**Sample Similarity:** Measure the similarity between individual samples, either within the same class or across the entire dataset.

To achieve these goals, the project integrates the Bray-Curtis similarity metric with the CSG framework. This integration enables precise calculations of similarity both at the class level and at the sample level, 
providing a comprehensive understanding of the relationships within the dataset. These insights are particularly useful for assessing dataset quality, identifying redundant or overlapping data, and understanding 
potential challenges for machine learning models.



The first step is to install the library using pip as follows:
```pip install new_spectral_metric``` bash 

Once installed, you can access the files that make up the library. The library provides functionality to compute similarity metrics at different levels:

Intra-class similarity: Use new_estimator_intra_class (available within new_spectral_metric) to calculate the similarity between samples within the same class.
Dataset-wide similarity: Use new_estimator_all_samples to compute similarity across all samples in the dataset.
Alternatively, you can use new_estimator_complete, which combines both intra-class similarity and dataset-wide similarity into a single estimator for comprehensive analysis.
