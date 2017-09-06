# The One Hundred Layers Tiramisu
Image Segmentation using CNN based on paper titled "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation" (https://arxiv.org/abs/1611.09326)

Note: Dataset used is CamVid dataset and was not uploaded due to the size. I splitted the dataset into training(450 images), validation(125 images), and test(76 images) sets. Things were slightly changed compared to the paper:
1. Average pooling instead of max pooling
2. Upsampling rather than deconvolution
3. Adam instead of RMSprop
4. No data augmentation
5. Images were resized to 256 x 384

I got 92.79% accuracy on the test set. Here is the results

<p align="center">
  <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/tiramisu_segmentation/images/results.png">
</p>

Note: TensorFlow and PyTorch implementations are coming soon.