# Adversarial Examples Generator Network
I was working on this project but then I realized something similar was already submitted to arXiv (https://arxiv.org/abs/1703.09387).The only difference from the one proposed in the paper is that the network that I created does not try to minimally modify the classifier output and I also did not implement the reranking function (e.g. we want to make exceptions if the fake target and actual target are the same). This is essentially a generative model that generates adversarial examples to fool a trained neural network classifier.

<p align="center">
  <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/adversarial_transformation_networks/diagrams/adversarial-examples-generator-network.jpg">
</p>

The idea is to let the network **learns to generate images that look like the images in the training/test set, but the generated images will be misclassified by a trained classifier as desired fake classes with high confidence** (e.g. a network can be trained to generate images that look like MNIST dataset but the images will always be misclassified as '0' with high confidence by the trained neural network classifier)

Notebooks:
1. Attacking Simple CNN MNIST Classifier with Adversarial Examples Generator Network
    
    Summary: Using Adversarial Examples Generator Network to attack simple classifier made of convolutional neural network that has 99.39% accuracy on MNIST test set. The generative network was able to fool the classifier to the point where the classifier accuracy went down to 9.84%, and majority of the predictions are predicted with high confidence.

    <p align="center">
    <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/adversarial_transformation_networks/images/notebook1.png">
    </p>

2. Attacking Simple CNN CIFAR10 Classifier with Adversarial Examples Generator Network
    
    Summary: Using Adversarial Examples Generator Network to attack simple CIFAR10 classifier made of convolutional neural network that has 77.61% accuracy on CIFAR10 test set. The generative network was able to fool the classifier to the point where the classifier accuracy went down to 17.57%.

    <p align="center">
    <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/adversarial_transformation_networks/images/notebook2.png">
    </p>
