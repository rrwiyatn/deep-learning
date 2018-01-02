# Generating Adversarial Examples Using Fast Gradient Sign Method (FGSM)
Generating adversarial examples using fast gradient sign method based on paper: https://arxiv.org/abs/1412.6572

Here are some samples of generated adversarial images and the classifier's confidence score:

<p align="center">
  <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/fast_gradient_sign_attack/results/results.png">
</p>

Fig.1: The top row shows the original images from MNIST with the confidence score of the classifier. Middle row shows the perturbations, and the bottom row shows the adversarial MNIST images with the confidence score of the classifier when asked to classify the adversarial images.

The accuracy of the classifier on the MNIST test set was: 98.48%

The accuracy of the classifier on the adversarial MNIST test set was: 4.98%
