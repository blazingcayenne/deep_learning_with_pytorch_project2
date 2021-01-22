# deep_learning_with_pytorch_project2

## Introduction

This work is the second project of the OpenCV Pytroch Course's. Its focus is image classification. Half of the score is based on implementation. The other half is based on model performance.

### Data Description

The dataset consists of 8,174 images in 13 Kenyan food type classes. Sample images of KenyanFood13 dataset and the number of images in each of the classes are shown below:

<center>

<img src="https://github.com/monajalal/Kenyan-Food/blob/master/img/KenyanFood13.png?raw=true" alt="Sample images" width="600px" /><br>
**Figure 1:** A sample image from each class of the KenyanFood13 dataset.<br>

</center>

The data was split into public train set and private test set which is used for evaluation of submissions. The public subset should be split into train and validation sets.

### Goal

To create a model that predicts a type of food represented on each image in the KenyanFood13 dataset's private test set. Pre-trained models may be used. The performance metric is accuracy. To receive any performance point, an accuracy of 70% must be achieved. To receive full points, an accuracy of 75% must be achieved.

### Approach

Rather than create my own CNN architecture, I will explore transfer learning of PyVision models that have been trained on the ImageNet data set. Pedro Marcelino defines three strategies for fune-tuning pretrained models in **[Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)** as depicted below.

<center>

![Fine-tuning strategies](https://miro.medium.com/max/700/1*9t7Po_ZFsT5_lZj445c-Lw.png)<br>
**Figure 2:** Three strategies for fine-tuning pre-trained models.

</center>

Marcelino gives guidance as to which strategy to use based on one's dataset size and similarity (see Figure 3). Marcelino defines a small dataset as one with less than 1000 images per class. According to this definition, the KenyanFood13 dataset is small. Regarding dataset similarity, Marceline states:

> ...  let common sense prevail. For example, if your task is to identify cats and dogs, ImageNet would be a similar dataset because it has images of cats and dogs. However, if your task is to identify cancer cells, ImageNet can’t be considered a similar dataset.

Fortunately, the ImageNet dataset contains 10 food classes: apple, banana, broccoli, burger, egg, french fries, hot dog, pizza, rice, and strawberry. Unfortunately, ImageNet's food images look significantly different from the images in Kenyan13Food dataset. Hence, I would say the dataset similarity is low to moderate.

<center>

<img src="https://miro.medium.com/max/510/1*7ZD-u-h8hFPuN2PYJvLMBw.png" alt="Decision map" width="400px" /><br>
**Figure 3:** Decision map for fine-tuning pre-trained models.

</center>

According to the decision map, pre-trained models should be fine tuned by either freezing part or all of the convolutional base.

Dishashree Gupta also defines strategies for fine-tuning pre-trained models in his blog post, **[Transfer learning and the art of using Pre-trained Models in Deep Learning](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)**. Gupta's strategies, while similar to Marcelino, have the following differences.

* Large datasets with low similarity should train models from scratch.
* Large datasets with high similarity should train the entire pre-trained model.

### Implementation Overview

While my primary objective is to fine-tune a pre-trained model to achieve a minimum of 75% accuracy on the KenyanFood13 dataset, my secondary objective is to improve my proficiency with the [Python](https://www.python.org/) computer programming language. (Prior to this class, my Python expertise was almost non-existent.) Consequently, I developed class hierachies that enable the rapid implementation of fine-tuning experiments on following pre-trained TorchVision models using any of Marcelino or Gupta's strategies.

* ResNet-18
* ResNet-34
* ResNet-50
* ResNet-101
* ResNet-152
* ResNeXt-50-32x4d
* ResNeXt-101-32x8d
* Wide ResNet-50-2
* Wide ResNet-101-2
* VGG-11 with batch normalization
* VGG-13 with batch normalization
* VGG-16 with batch normalization
* VGG-19 with batch normalization
* DenseNet-121
* DenseNet-169
* DenseNet-201
* DenseNet-161

I used and modified the Trainer Pipeline module introduced in **Week 6 - Best Practicing in Deep Learning > How to structure your Project for Scale**. Modications to the trainer module include, but are not limited to, the following:

* additional configuration parameters, 
* saving the model state only when the average loss on the validation set reaches a new low, 
* prematurely stopping training when either the loss or accuracy does not significantly improve over a certain number of epochs, and
* extending the visualization base and TensorBoard classes to support the logging of images, figures, graphs, and PR curves.

Experiments are identified by prefix "Exp" followed by three uppercase capital letters per the regular expression \(Exp\[A-Z\]\[A-Z\]\[A-Z\]\). The first and second letters designate the experiment group and experiment set respectiviely. The last letter designates an individual experiment. Hence, all experiments that begin with "ExpA" belong to Group A, while all experiments that begin with "ExpAB" belong to Group A, Set B.

I implemented the following groups of experiments.

* Group A to explore the data and verify the training pipeline.
* Group B to explore all fine-tuning strategies on models from the ResNet, VGG, and DenseNet families.
* Group C to explore optimizing a few of Group B experiments that yielded the highest accuracy with over-fitting the training data.
* Group D to explore whether an ensemble of models with different architectures yields a higher accuracy than its constitutes.
* Group E to explore miscellaneous issues, e.g., performance of Project 1 model, normalizing KenyanFood13 data by its mean/std, training on grayscale images, impact of no or poor data augmentation, impact of learning rate when traing unfrozen pre-trained convolutional layers.

## Experiment Group A: Data Visualization Experiments and Training Pipeline Check

### Summary

This first set of experiments log contact sheets, 6 x 6 grids of images, of each food type to the visualizer with and without data augmentation. 

The second set of experiments train only the classifier layer (fc) of the pre-trained Resnet18 model using a subset of the data without augmentation to check the training pipeline. In the first experiment, training will stop after 40 epochs. In the second experiment, training will stop after 40 epochs or when the smoothed accuracy (computed by [expontential moving average](https://towardsdatascience.com/moving-averages-in-python-16170e20f6c#144e) with an alpha = 0.3) does not decrease by 2% within 10 epochs.

The third set of experiments vary the number of data loader worker threads to determine the optimal number for future experiments. These experiments stop after 11 epochs. The time between logging the first and eleventh epochs' test metrics divided by 10 will be used to evaluate data loading efficiency. Since the purpose to data koading, not the forward/back-propogation cycle, a simpe model, ResNet18, was used. Furthermore, saving the model's state is disabled to eliminate its time contribution.

### Results

I used the data augmentation transforms I created for project 1. The data validation experiment revealed issues with these transforms. First, the color jitter transform dramatically changed the image's color. While this was not detrimental to classify cats, dogs, and pandas; I suspect it may reduce accuracy on the KenyanFood13 dataset. I will test this theory in the last group of experiments after the assignment has been completed. Second, the amount of translation and scaling was too agressive.

To properly set the data augmentation transform parameters, I ran several experiments not shown in this notebook. These experiments disabled all but one type of augmentation in order to "tune" it. For example, to properly set the hue parameter of the color jitter transform, I disabled the horizontal/vertical flips, affine, and erase transforms. Furthermore, I set the color jitter's brightness, contrast, and saturation to the values that would produce the original image. I then found acceptable minimum and maximum values for the hue parameter. After conducting all of these data augmentation tuning experiments, I updated the configuration file and re-ran the data visualization experiment. I visualized the entire dataset to external files, but only logged the following 6 x 6 contact sheets to Tensorboard.
* ExpAAA - Original Images
* ExpAAA - "Tuned" Augmented Images
* ExpAAB - "Untuned" Augmented Images

The training pipeline check experiment performed as expected. The training and test loss decreased and the accuracy increased to approximtely 60%.

The results of the data loader experiments are shown below. For a small number of data loader threads, adding an additional thread significantly increases the data loader's efficiency allow it to keep up with GPU. However, past six or seven threads, the impact is minimal on a small network. Since I am using a workstation with 8 hyper-threaded cores for a total of 16 CPUs and GPU acceleration, I dedicated 6 to 12 worker threads to the data loader. As the model complexity increases, the impact of fewer worker threads lessens because each worker has more time to load and transform images before the GPU requires the next batch.

<center>

**Table 1.** The impact of the number of data loader worker threads on a training cycle.
|Experiment|Workers|Time/Epoch|
|:---:|:---:|:---:|
|ExpACA|1|02:22|
|ExpACB|2|01:16|
|ExpACC|3|00:53|
|ExpACD|4|00:41|
|ExpACE|5|00:33|
|ExpACF|6|00:28|
|ExpACG|7|00:25|
|ExpACH|8|00:23|
|ExpACI|9|00:23|
|ExpACJ|10|00:22|
|ExpACK|11|00:22|
|ExpACL|12|00:21|
|ExpACM|13|00:21|
|ExpACN|14|00:20|
|ExpACO|15|00:19|
|ExpACP|16|00:19|

</center>

## Experiment Group B: Exhaustive Exploration

This group of experiments ignores the strategies given by [Marcelino](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751) and [Gupta](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/). Rather than select a couple pre-trained models and fully or partially freeze the convolution base, these experiments are exhaustive. They explore all all (implemented) pretrained models and freezing options. Since I have nearly unlimited access to a Linux workstation with a Intel Core i7-9800X CPU, 64 GB RAM, and a GeForce RTX 2080 Ti GPU; I decided to run these experiments. My motivation is to explore:

* transfer learning between different model families,
* transfer learning between different models within a family,
* transfer learning compared to learning from scratch, and
* transfer leearning as convolution blocks are progressively unfrozen.

These experiments would be a waste of time for someone with experience in transfer learning. However, for a beginner with a couple of GPU days to spare, they may provide insight.

### Summary

This group of experiments explores transfer learning. Each set of experiments in this group explores transfer learning on a specific model and conducts the following experiments. 
* ExpB?A - Train the entire untrained model
* ExpB?B - Train the pretrained model's classifier layer (tuning_level = 0)
* ExpB?C - Train the pretrained model's classifier layer and last convolution block (tuning_level = 1)
* ExpB?D - Train the pretrained model's classifier layer and last 2 convolution blocks (tuning_level = 2)
* ExpB?E - Train the pretrained model's classifier layer and last 3 convolution blocks (tuning_level = 3)
* ExpB?F - Train the pretrained model's classifier layer and last 4 convolution blocks (tuning_level = 4)
* ExpB?G - Train the entire pretrained model

Preliminary tests indicate that overfitting is possible with the large models even when the KenyanFood13 images are significantly augmented. My first inclination, which I rejected, was to explore transfer learning on representatives from the ResNet, VGG, and DenseNet model families. My selection criterion was the model with the lowest ImageNet Top-1 error. However, these representatives were significantly more complex than their siblings, so overfitting is likely. Consequently, I will perform transfer learning experiments on every implemented models (see below).
* ResNet-18
* ResNet-34
* ResNet-50
* ResNet-101
* ResNet-152
* ResNeXt-50-32x4d
* ResNeXt-101-32x8d
* Wide ResNet-50-2
* Wide ResNet-101-2
* VGG-11 with batch normalization
* VGG-13 with batch normalization
* VGG-16 with batch normalization
* VGG-19 with batch normalization
* DenseNet-121
* DenseNet-169
* DenseNet-201
* DenseNet-161

Training will stop after 100 epochs or when the smoothed accuracy (computed by expontential moving average with an alpha = 0.3) does not decrease by 1% within 10 epochs.

### Results

The following table summarizes the experiments. The displayed accuracy is the percentage of images the model correctly classified from the test set at the epoch where the test loss was lowest. The overfitting metric is computed as following.

* The test loss is divided by the training loss at each epoch.
* The best fit line is computed for the above data.
* The slope is chosen as the overfitting metric.

An overfitting value of zero indicates the test loss declines at the same rate as the training loss; hence, no overfitting. 

_ToDo: Show three different loss plots._

<center>

**Table 2.** The accuracy and overfitting metric for each experiment.
|Experiment|Accuracy|Overfitting Metric|
|:---|:---:|:---:|
|BAA-ResNet18|41.11|0.001|
|BAB-ResNet18-PT0|58.50|0.000|
|BAC-ResNet18-PT1|69.13|0.005|
|BAD-ResNet18-PT2|72.18|0.010|
|BAE-ResNet18-PT3|72.87|0.011|
|BAF-ResNet18-PT4|72.80|0.012|
|BAG-ResNet18-PT5|72.88|0.012|
|BBA-ResNet34|40.80|0.000|
|BBB-ResNet34-PT0|58.43|0.001|
|BBC-ResNet34-PT1|69.05|0.007|
|BBD-ResNet34-PT2|73.80|0.016|
|BBE-ResNet34-PT3|74.64|0.019|
|BBF-ResNet34-PT4|73.87|0.020|
|BBG-ResNet34-PT5|73.56|0.020|
|BCA-ResNet50|31.69|0.000|
|BCB-ResNet50-PT0|60.95|0.000|
|BCC-ResNet50-PT1|74.07|0.012|
|BCD-ResNet50-PT2|76.59|0.022|
|BCE-ResNet50-PT3|77.74|0.025|
|BCF-ResNet50-PT4|77.00|0.026|
|BCG-ResNet50-PT5|77.12|0.026|
|BDA-ResNet101|25.32|0.000|
|BDB-ResNet101-PT0|61.25|0.001|
|BDC-ResNet101-PT1|73.33|0.012|
|BDD-ResNet101-PT2|76.61|0.039|
|BDE-ResNet101-PT3|76.38|0.040|
|BDF-ResNet101-PT4|77.59|0.039|
|BDG-ResNet101-PT5|76.91|0.041|
|BEA-ResNet152|26.56|0.000|
|BEB-ResNet152-PT0|59.04|0.001|
|BEC-ResNet152-PT1|72.48|0.009|
|BED-ResNet152-PT2|77.21|0.036|
|BEE-ResNet152-PT3|78.22|0.038|
|BEF-ResNet152-PT4|77.75|0.039|
|BEG-ResNet152-PT5|78.45|0.037|
|BFA-ResNeXt50|35.51|0.000|
|BFB-ResNeXt50-PT0|62.55|0.000|
|BFC-ResNeXt50-PT1|74.48|0.025|
|BFD-ResNeXt50-PT2|78.59|0.047|
|BFE-ResNeXt50-PT3|78.51|0.050|
|BFF-ResNeXt50-PT4|78.21|0.053|
|BFG-ResNeXt50-PT5|78.67|0.051|
|BGA-ResNeXt101|34.66|-0.001|
|BGB-ResNeXt101-PT0|64.06|0.001|
|BGC-ResNeXt101-PT1|75.07|0.039|
|BGD-ResNeXt101-PT2|77.90|0.117|
|BGE-ResNeXt101-PT3|78.12|0.121|
|BGF-ResNeXt101-PT4|77.90|0.130|
|BGG-ResNeXt101-PT5|77.28|0.127|
|BHA-WideResNet50|34.11|0.000|
|BHB-WideResNet50-PT0|59.14|0.000|
|BHC-WideResNet50-PT1|73.40|0.020|
|BHD-WideResNet50-PT2|77.31|0.030|
|BHE-WideResNet50-PT3|77.31|0.032|
|BHF-WideResNet50-PT4|77.93|0.032|
|BHG-WideResNet50-PT5|78.15|0.032|
|BIA-WideResNet101|29.02|-0.001|
|BIB-WideResNet101-PT0|56.86|0.000|
|BIC-WideResNet101-PT1|74.48|0.018|
|BID-WideResNet101-PT2|77.37|0.039|
|BIE-WideResNet101-PT3|77.46|0.042|
|BIF-WideResNet101-PT4|77.77|0.044|
|BIG-WideResNet101-PT5|77.84|0.040|
|BJA-VGG11BN|38.45|0.003|
|BJB-VGG11BN-PT0|61.54|0.003|
|BJC-VGG11BN-PT1|69.20|0.006|
|BJD-VGG11BN-PT2|72.79|0.011|
|BJE-VGG11BN-PT3|73.63|0.012|
|BJF-VGG11BN-PT4|73.69|0.013|
|BJG-VGG11BN-PT5|73.62|0.013|
|BKA-VGG13BN|40.88|0.001|
|BKB-VGG13BN-PT0|62.80|0.002|
|BKC-VGG13BN-PT1|71.27|0.005|
|BKD-VGG13BN-PT2|73.55|0.011|
|BKE-VGG13BN-PT3|73.48|0.013|
|BKF-VGG13BN-PT4|73.78|0.015|
|BKG-VGG13BN-PT5|74.16|0.015|
|BLA-VGG16BN|38.67|0.001|
|BLB-VGG16BN-PT0|62.58|0.002|
|BLC-VGG16BN-PT1|72.51|0.010|
|BLD-VGG16BN-PT2|76.01|0.019|
|BLE-VGG16BN-PT3|76.09|0.021|
|BLF-VGG16BN-PT4|75.87|0.022|
|BLG-VGG16BN-PT5|76.24|0.022|
|BMA-VGG19BN|38.72|0.001|
|BMB-VGG19BN-PT0|60.66|0.002|
|BMC-VGG19BN-PT1|73.56|0.009|
|BMD-VGG19BN-PT2|76.99|0.018|
|BME-VGG19BN-PT3|77.60|0.020|
|BMF-VGG19BN-PT4|76.76|0.020|
|BMG-VGG19BN-PT5|76.69|0.020|
|BNA-DenseNet121|37.48|0.001|
|BNB-DenseNet121-PT0|62.00|0.002|
|BNC-DenseNet121-PT1|69.97|0.008|
|BND-DenseNet121-PT2|74.61|0.019|
|BNE-DenseNet121-PT3|75.76|0.025|
|BNF-DenseNet121-PT4|75.30|0.025|
|BNG-DenseNet121-PT5|75.93|0.025|
|BOA-DenseNet169|38.45|0.001|
|BOB-DenseNet169-PT0|63.62|0.002|
|BOC-DenseNet169-PT1|71.33|0.016|
|BOD-DenseNet169-PT2|75.60|0.032|
|BOE-DenseNet169-PT3|77.44|0.037|
|BOF-DenseNet169-PT4|77.06|0.038|
|BOG-DenseNet169-PT5|78.20|0.038|
|BPA-DenseNet201|37.35|0.001|
|BPB-DenseNet201-PT0|63.22|0.001|
|BPC-DenseNet201-PT1|72.54|0.009|
|BPD-DenseNet201-PT2|78.21|0.031|
|BPE-DenseNet201-PT3|78.89|0.036|
|BPF-DenseNet201-PT4|79.33|0.037|
|BPG-DenseNet201-PT5|79.43|0.035|
|BQA-DenseNet161|41.78|0.001|
|BQB-DenseNet161-PT0|67.37|0.002|
|BQC-DenseNet161-PT1|74.09|0.015|
|BQD-DenseNet161-PT2|77.13|0.034|
|BQE-DenseNet161-PT3|78.35|0.040|
|BQF-DenseNet161-PT4|78.12|0.041|
|BQG-DenseNet161-PT5|77.67|0.041|

</center>

## Experiment Group C: ...

### Summary

### Results

|Experiment|Accuracy|Overfitting Metric|
|:---|:---:|:---:|
|CAA-ResNeXt101-LR1E-3|77.59|0.313|
|CAB-ResNeXt101-LR5E-4|77.90|0.193|
|CAC-ResNeXt101-LR1E-4|78.57|0.027|
|CAD-ResNeXt101-LR5E-5|78.58|0.010|
|CAE-ResNeXt101-LR1E-5|72.93|-0.001|
|CBA-VGG19BN-LR1E-3|75.40|0.063|
|CBB-VGG19BN-LR5E-4|76.38|0.034|
|CBC-VGG19BN-LR1E-4|76.85|0.006|
|CBD-VGG19BN-LR5E-5|74.63|0.002|
|CBE-VGG19BN-LR1E-5|66.55|-0.001|
|CCA-DenseNet161-LR1E-3|76.82|0.175|
|CCB-DenseNet161-LR5E-4|77.29|0.090|
|CCC-DenseNet161-LR1E-4|76.75|0.010|
|CCD-DenseNet161-LR5E-5|76.37|0.003|
|CCE-DenseNet161-LR1E-5|68.85|-0.001|

## Experiment Group D: ...

### Summary

### Results

|Experiment|Accuracy|Overfitting Metric|
|:---|:---:|:---:|
|DAA-EfficientNet-B0-PT2|74.00|0.003|
|DAB-EfficientNet-B1-PT2|76.75|0.004|
|DAC-EfficientNet-B2-PT2|77.59|0.005|
|DBA-EfficientNet-B0-PT5|74.92|0.008|
|DBB-EfficientNet-B1-PT5|78.89|0.010|
|DBC-EfficientNet-B2-PT5|79.28|0.011|