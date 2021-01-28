# deep_learning_with_pytorch_project2

## Introduction

This work is the second project of the OpenCV Pytroch Course's. Its focus is image classification. Half of the score is based on implementation. The other half is based on model performance.

### Data Description

The dataset consists of 8,174 images in 13 Kenyan food type classes. Sample images of KenyanFood13 dataset and the number of images in each of the classes are shown below:

<center>

![Sample images from the KenyanFood13 Dataset](https://media.githubusercontent.com/media/blazingcayenne/deep_learning_with_pytorch_project2/main/images/KenyanFood13.jpg?raw=true)<br>
**Figure 1:** A sample image from each class of the KenyanFood13 dataset.

</center>

The data was split into public training set and private test set which is used for evaluation of submissions. The public set can be split into training and validation subsets.

### Goal

To create a model that predicts a type of food represented on each image in the KenyanFood13 dataset's private test set. Pre-trained models may be used. The performance metric is accuracy. To receive any performance point, an accuracy of 70% must be achieved. To receive full points, an accuracy of 75% must be achieved.

### Approach

Rather than create my own CNN architecture, I will explore transfer learning of PyVision models that have been trained on the ImageNet data set. Pedro Marcelino defines three strategies for fine-tuning pretrained models in **[Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)**.

* Train the entire model
* Train some layers and leave others frozen
* Train only the classifer by freezing the convolutional base

Marcelino gives guidance as to which strategy to use based on one's dataset size and similarity (see Figure 2). Marcelino defines a small dataset as one with less than 1000 images per class. According to this definition, the KenyanFood13 dataset is small. However, it is unclear whether the application of data augmentation would reclassify the size of this class. Hence, this project will explore training the entire pretrained model as well as training some layers and leaving others frozen. Regarding dataset similarity, Marceline states:

> ...  let common sense prevail. For example, if your task is to identify cats and dogs, ImageNet would be a similar dataset because it has images of cats and dogs. However, if your task is to identify cancer cells, ImageNet can’t be considered a similar dataset.

Fortunately, the ImageNet dataset contains 10 food classes: apple, banana, broccoli, burger, egg, french fries, hot dog, pizza, rice, and strawberry. Unfortunately, ImageNet's food images look significantly different from the images in Kenyan13Food dataset. Hence, the dataset similarity is low to moderate. Nevertheless, this project will explore freezing the convolutional base and training only the classifier.

<center>

![Decision map](https://media.githubusercontent.com/media/blazingcayenne/deep_learning_with_pytorch_project2/main/images/TransferLearningApproaches.png?raw=true)<br>
**Figure 2:** Decision map for fine-tuning pre-trained models.

</center>

According to the decision map, pre-trained models should be fine tuned by either freezing part or all of the convolutional base. This project will test the efficiency of this guidance.

Dishashree Gupta also defines strategies for fine-tuning pre-trained models in his blog post, **[Transfer learning and the art of using Pre-trained Models in Deep Learning](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)**. Gupta's strategies, while similar to Marcelino, have the following differences.

* Large datasets with low similarity should train models from scratch.
* Large datasets with high similarity should train the entire pre-trained model.

### Implementation Overview

While my primary objective is to fine-tune a pre-trained model to achieve a minimum of 75% accuracy on the KenyanFood13 dataset, my secondary objective is to improve my proficiency with the [Python](https://www.python.org/) computer programming language. (Prior to this class, my Python expertise was almost non-existent.) Consequently, I developed class hierachies that enable the rapid implementation of fine-tuning experiments on following pre-trained TorchVision and EfficientNet models using any of Marcelino or Gupta's strategies.

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
* EfficientNet-B0
* EfficientNet-B1
* EfficientNet-B2
* EfficientNet-B3
* EfficientNet-B4
* EfficientNet-B5
* EfficientNet-B6
* EfficientNet-B7

I used and modified the Trainer Pipeline module introduced in **Week 6 - Best Practicing in Deep Learning > How to structure your Project for Scale**. Modications to the trainer module include, but are not limited to, the following:

* additional configuration parameters,
* saving the model state only when the average loss on the validation set reaches a new low,
* prematurely stopping training when either the loss or accuracy does not significantly improve over a certain number of epochs, and
* extending the visualization base and TensorBoard classes to support the logging of images, figures, graphs, and PR curves.

Experiments are identified by three uppercase capital letters per the regular expression \(\[A-Z\]\[A-Z\]\[A-Z\]\). The first and second letters designate the experiment group and experiment set respectiviely. The last letter designates an individual experiment. Hence, all experiments that begin with "A" belong to Group A, while all experiments that begin with "AB" belong to Group A, Set B.

I implemented the following groups of experiments.

* Group A to explore the data and verify the training pipeline.
* Group B to explore the impact of learning rate on training pretrained EfficientNet models.
* Group C to explore the impact of EfficientNet model complexity on bias and variance.

## Experiment Group A: Data Visualization and Training Pipeline Check

### Introduction

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

The results of the data loader experiments are shown below. For a small number of data loader threads, adding an additional thread significantly increases the data loader's efficiency allow it to keep up with GPU. However, past six or seven threads, the impact is minimal on a small network. Since I am using a workstation with 8 hyper-threaded cores for a total of 16 CPUs and GPU acceleration, I dedicated 8 to 12 worker threads to the data loader. As the model complexity increases, the impact of fewer worker threads lessens because each worker has more time to load and transform images before the GPU requires the next batch.

<center>

**Table A1.** The impact of the number of data loader worker threads on a training cycle.
|Experiment|Workers|Time/Epoch|
|:---:|:---:|:---:|
|ACA|1|02:23|
|ACB|2|01:16|
|ACC|3|00:53|
|ACD|4|00:40|
|ACE|5|00:33|
|ACF|6|00:28|
|ACG|7|00:25|
|ACH|8|00:22|
|ACI|9|00:22|
|ACJ|10|00:22|
|ACK|11|00:21|
|ACL|12|00:21|
|ACM|13|00:20|
|ACN|14|00:19|
|ACO|15|00:18|
|ACP|16|00:18|

</center>
<br>

## Experiment Group B: Learning Rate

### Introduction

The purpose of these experiments is to explore the impact of learning rate on training the following pretrained models.

* EfficientNet-B0 (Set A)
* EfficientNet-B2 (Set B)
* EfficientNet-B4 (Set C)

The entire network was trained, i.e., no layers were frozen. Each model was trained for 10 epochs with learning rates that varied between 1.0E-06 and 1.0E-02 with 4 learning rates per decade.

### Results

For each set of experiments, the following metrics were plotted as a function of learning rate in Figures B1, B2, and B3.

* Minimum test loss (min loss)
* Area under the test loss curve (auc loss)
* Test accuracy at the epoch of minimum test loss (accuracy)
<br>
<br>

<center>

![EfficientNet-B0 learning rate results](https://media.githubusercontent.com/media/blazingcayenne/deep_learning_with_pytorch_project2/main/images/BA__EfficientNetB0_PT5.png?raw=true)<br>
**Figure B1.** The impact of learning rate on the EfficientNet-B0 model.<br>
<br>

![EfficientNet-B2 learning rate results](https://media.githubusercontent.com/media/blazingcayenne/deep_learning_with_pytorch_project2/main/images/BB__EfficientNetB2_PT5.png?raw=true)<br>
**Figure B1.** The impact of learning rate on the EfficientNet-B2 model.<br>
<br>

![EfficientNet-B4 learning rate results](https://media.githubusercontent.com/media/blazingcayenne/deep_learning_with_pytorch_project2/main/images/BC__EfficientNetB4_PT5.png?raw=true)<br>
**Figure B1.** The impact of learning rate on the EfficientNet-B4 model.<br>
<br>


</center>

Table B1 enumerates the aforementioned metrics for each set's experiment with the lowest test loss.
<br>
<br>

<center>

**Table B1.** The impact of learning rate on the EfficientNet models.
|Experiment|Learning Rate|Min Loss|AUC Loss|Accuracy|
|:---|:---:|:---:|:---:|:---:|
|BAK_EfficientNetB0_PT5_LR3E-4|3.16e-04|0.76|9.69|75.22|
|BBI_EfficientNetB2_PT5_LR1E-4|1.00e-04|0.69|8.51|77.63|
|BCH_EfficientNetB4_PT5_LR6E-5|5.62e-05|0.57|6.80|81.88|

</center>
<br>

## Experiment Group C: Model Complexity

