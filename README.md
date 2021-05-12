# COVID-19 detection with lung segmentation
 We describe these deep neural networks in the paper: "COVID-19 detection using chest X-rays: is lung segmentation important for generalization?",
 available at https://arxiv.org/abs/2104.06176v1.

# Abstract: 
We evaluated the generalization capability of deep neural networks (DNNs), trained to classify chest X-rays as COVID-19, normal or pneumonia, using a relatively small and mixed dataset.
We proposed a DNN architecture to perform lung segmentation and classification. It stacks a segmentation module (U-Net), an original intermediate module and a classification module (DenseNet201). We compared it to a DenseNet201.
To evaluate generalization, we tested the DNNs with an external dataset (from distinct localities) and used Bayesian inference to estimate the probability distributions of performance metrics, like F1-Score.
Our proposed DNN achieved 0.917 AUC on the external test dataset, and the DenseNet, 0.906. Bayesian inference indicated mean accuracy of 76.1% and [0.695, 0.826] 95% HDI with segmentation and, without segmentation, 71.7% and [0.646, 0.786].
We proposed a novel DNN evaluation technique, using Layer-wise Relevance Propagation (LRP) and the Brixia score. LRP heatmaps indicated that areas where radiologists found strong COVID-19 symptoms and attributed high Brixia scores are the most important for the stacked DNN classification.
External validation showed smaller accuracies than internal validation, indicating dataset bias, which segmentation reduces. Performance in the external dataset and LRP analysis suggest that DNNs can be trained in small and mixed datasets and detect COVID-19.
