# NEURAL STYLE TRANSFER

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: MD SAMIR AKHTAR

**INTERN ID**: CTIS9348

**DOMAIN**: ARTIFICIAL INTELLIGENCE

**DURATION**: 8 WEEKS

**MENTOR**: NEELA SANTOSH

## Introduction

Neural Style Transfer (NST) is one of the most visually captivating applications of Convolutional Neural Networks (CNNs). At its core, NST is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image. This process was first introduced by Leon Gatys et al. in their seminal 2015 paper, "A Neural Algorithm of Artistic Style," which demonstrated that deep neural networks could separate and recombine the content and style of arbitrary images.

The fundamental breakthrough of NST lies in the realization that the high-level features of a pre-trained CNN can act as a proxy for the "content" of an image, while the correlations between these features can represent its "style." To achieve this, a network like VGG-19, pre-trained on the massive ImageNet dataset, is typically used. In the early layers of the network, the neurons respond to simple edges and textures. However, as we go deeper into the network, the neurons respond to increasingly complex structures—shapes, objects, and semantic concepts. By extracting the activations from these deeper layers, we can capture the structural arrangement and objects within an image, which constitutes the "content."

Style, on the other hand, is more elusive. Gatys and his team discovered that style could be captured by looking at the correlations between different filter responses in a given layer. This is mathematically represented by a Gram Matrix. A Gram Matrix calculates the inner product between the vectorized feature maps of a layer. By doing this, it captures which features tend to activate together across the entire image, effectively ignoring the spatial arrangement (the content) and focusing on the statistical distribution of textures, colors, and patterns. This statistical representation is what we define as the "style" of the image.

The process of Neural Style Transfer is framed as an optimization problem. We start with a target image—which can be a copy of the content image, the style image, or even white noise—and iteratively modify its pixel values to minimize a multi-part loss function. This total loss function consists of three primary components: the content loss, the style loss, and the total variation loss. The content loss ensures that the target image maintains the structural integrity of the original content image. The style loss ensures that the target image adopts the textures and color schemes of the style image. Finally, the total variation loss acts as a regularizer, encouraging spatial smoothness and reducing high-frequency noise in the generated image.

Mathematically, we use gradient descent (or more advanced optimizers like L-BFGS) to backpropagate the gradients from the loss functions all the way back to the input image pixels, rather than the network weights. Over several hundred or thousand iterations, the input image gradually transforms. It begins to exhibit the brushstrokes of Van Gogh, the geometric patterns of Picasso, or the vibrant colors of a sunset, all while retaining the recognizable shapes of the original photograph.

The significance of Neural Style Transfer extends beyond just creating "cool" filters. It has paved the way for deeper research into the interpretability of neural networks, helping scientists understand what different layers of a CNN are actually "seeing." It has also influenced the development of generative models, including GANs (Generative Adversarial Networks) and diffusion models, which are now at the forefront of the AI art revolution. Today, NST is used in everything from mobile photo-editing apps to professional film post-production, proving that the intersection of mathematics, computer science, and art can yield results that are both intellectually profound and aesthetically breathtaking.

## Project Structure
- `Neural-Style-Transfer.py`: The main Python script containing the implementation.
- `content.png`: Sample content image.
- `style.png`: Sample style image.
- `landscape_content.png`: Additional content image.
- `new_content.png`: Additional content image.

## How to Run
1. Ensure you have Python and the required libraries (TensorFlow, NumPy, PIL) installed.
2. Run the script:
   ```bash
   python Neural-Style-Transfer.py
   ```

## Output

<img width="1920" height="1020" alt="Image" src="https://github.com/user-attachments/assets/8dc66aa9-ae9d-43b4-be2d-a306f6391d04" />
