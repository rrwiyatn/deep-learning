# Photographic Image Synthesis with Cascaded Refinement Networks
Photographic image synthesis from segmentation map based on paper titled "Photographic Image Synthesis with Cascaded Refinement Networks" (https://arxiv.org/abs/1707.09405)

Note: Dataset used is CamVid dataset and was not uploaded due to the size. I splitted the dataset into training(450 images), validation(125 images), and test(76 images) sets. I  did not synthesize the diverse collection as explained in section 3.4 in the paper and I also only used 7 refinement modules rather than 9.

These are some generated images (from the test set) when using tanh as the activation in the last output layer and without batchnorm, compared to a similar model but without bactchnorm and tanh activation:

<p align="center">
  <img src="https://github.com/rrwiyatn/deeplearning-ai/blob/master/photo_image_synthesis_CRN/results/comparison.png">
</p>
