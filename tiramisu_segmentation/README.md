# The One Hundred Layers Tiramisu
Image Segmentation using CNN based on paper titled "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation" (https://arxiv.org/abs/1611.09326)

Note: Dataset used is CamVid dataset and was not uploaded due to the size. I splitted the dataset into training(450 images), validation(125 images), and test(76 images) sets. Architecture was slightly changed by using average pooling instead of max pooling, and using upsampling rather than deconvolution

I got 92.79% accuracy on the test set. Here is the results

<p align="center">
  <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/tiramisu_segmentation/images/results.png">
</p>
