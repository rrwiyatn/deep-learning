# content_sort
Content Based Image Sorting Using Convolutional Neural Network

Inspired by style transfer algorithm (Gatys et. al.), I realized that we can extract the feature maps produced by the last layer of convolutional layer in VGG16 (pretrained with ImageNet weights) as the content representation of an image. Since the network is already pretrained for ImageNet, the network already learned representation of images for object recognition. 

I then used the network to get the average feature maps for each of the existing album. We can then perform content based image sorting by comparing the feature maps of the image that we want to sort with feature maps of the existing albums by just taking the sum squared error between them.

By using this method we can sort the images based on the users' preferences, and the algorithm should be able to learn users' preferences better as the number of images in each of the album increases.
