# CaptionGenie

CaptionGenie is an image captioning model that generates descriptive captions for images using a combination of a pre-trained VGG16 model for feature extraction and an LSTM-based neural network for caption generation. The model is trained on the Flickr8k dataset and produces captions that describe the content of images.



## Introduction
CaptionGenie is designed to understand images and generate accurate textual descriptions. The project combines techniques from computer vision and natural language processing, enabling machines to describe images in a human-like manner.

## Dataset
CaptionGenie is trained on the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which includes 8,000 images, each accompanied by its captions. The captions are preprocessed to remove special characters and standardize the text format.

## Model Architecture
CaptionGenie employs an autoencoder-like architecture:
1. **Feature Extractor (Encoder)**: A pre-trained VGG16 model is used to extract image features. The fully connected layers of the VGG16 model (before classification) are used to obtain feature vectors that represent the image.
  
2. **Caption Generator (Decoder)**: An LSTM-based model processes the image features and generates captions word by word. The model includes embedding layers, LSTM layers, and dense layers to predict the next word in the sequence.

## Results
CaptionGenie is capable of generating captions that accurately describe the content of images. While not always perfect, the model performs well on a variety of images.

## BLEU Scores
* **BLEU-1:** 0.521285
* **BLEU-2:** 0.302905

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or suggestions, feel free to reach out to us:

- **Email**: tyagiabhidyum@gmail.com
- **GitHub**: [Abhidyum](https://github.com/Abhidyum)






