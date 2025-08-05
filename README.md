# DeLaCruzBadiano1552

Preliminary website for analysis of the de la Cruz-Badiano codex of 1552

## Code and data  
All scripts to reproduce these analyses are found in `./analysis/scripts/`. Data to reproduce these analyses are found in `./analysis/data/`. The folders `./analysis/data/leaf_traces/`, `./analysis/data/texts/`, and `./analysis/data/texts_ES/` (leaf trace data and English and Spanish texts, respectively) will need to be unzipped before using. The FastText English and Spanish models `cc.en.300.bin` and `cc.es.300.bin` will need to be downloaded from this [link](https://fasttext.cc/docs/en/crawl-vectors.html) and placed into `./analysis/data/`. The pages of the document from which leaf traces are derived and on which they are plotted are derived from the public version of [The De la Cruz-Badiano Aztec Herbal of 1552](https://archive.org/details/aztec-herbal-of-1552) and the exact images found on [figshare](https://doi.org/10.6084/m9.figshare.29825189.v1) should be downloaded, the .zip file decompressed, and saved as the folder `pics` in `./analysis/data/`.

## Methods
The objective of this study was to analyze and classify Nahuatl plant names by integrating three distinct data modalities: linguistic, morphological, and structural. The data was derived from the 1939 William Gates English translation of the de la Cruz-Badiano Nahuatl Herbal. The pipeline consisted of unimodal analyses to characterize each data source, followed by a multi-modal approach to create a unified representation.

### Data Acquisition and Preprocessing
The primary data sources included digitized illustrations, corresponding English text descriptions, and a structural representation of plant name co-occurrence. For the morphological data, over 4,000 leaf outlines were hand-traced from illustrations using Fiji/ImageJ. These tracings were aligned using Generalized Procrustes Analysis (GPA) to normalize for position, scale, and rotation. The aligned shapes were then represented as a two-channel input for a convolutional neural network (CNN), with one channel being a radial Euler Characteristic Transform (ECT) in polar coordinates and the other a corresponding shape mask.

For the linguistic data, the English text from each plant's subchapter was aggregated. An automated Spanish translation was also generated for parallel analysis. FastText was employed to create word embeddings for both the English and Spanish texts. These embeddings were used for subsequent classification and clustering analyses.

A graph-based structural modality was created by constructing a network of Nahuatl plant names. Vertices in this graph represented unique Nahuatl names, and edges were weighted by the number of times two names co-occurred within the same subchapter. Node embeddings for this graph were then generated using the Node2Vec algorithm, with an embedding size of 128, a walk length of 30, and 200 walks per node.

### Data Augmentation
To address class imbalance and increase sample size for model training, a synthetic data generation method was used. After a Principal Component Analysis (PCA) was performed on the hand-traced leaf shapes, a Synthetic Minority Over-sampling Technique-like (SMOTE-like) approach was applied. Synthetic leaf shapes were created by sampling neighbors within the same class in the PCA space and linearly interpolating new shapes. These synthetic leaves were used to augment the training data for the image-based CNNs.

### Unimodal Analysis and Classification
Separate CNNs were trained to classify each modality into one of five main Nahuatl categories: the three main botanical categories (Xochitl, Quahuitl, Xihuitl) and two economic categories (Patli, Quilitl).

The image-based CNN used a two-channel input of radial ECT and shape mask to classify leaves. It was trained using stratified 5-fold cross-validation and employed ensemble prediction on the logits. To interpret the model's decisions, Grad-CAM visualizations were generated to highlight the image regions most relevant to classification.

A similar CNN architecture was adapted to classify the text data. This model took the FastText embeddings as input and was trained to classify the corresponding Nahuatl plant names. The key features distinguishing the classes were visualized using word clouds, which highlighted the most frequent and unique terms associated with each category.

### Multimodal Analysis and Integration
A three-tower CNN was developed to create a unified embedding space where the representations of all three modalities for the same plant were aligned. The model consisted of three parallel towers: an image tower (a 2D CNN), a text tower (a 1D CNN), and a graph tower (a linear layer). Each tower projected its modality's data into a shared 128-dimensional embedding space. The training objective was to minimize the distance between the embeddings of corresponding samples from all three modalities.

The model was trained using a combination of real and synthetic data within a stratified 5-fold cross-validation framework. A custom MultiModal Contrastive Loss function was used to simultaneously bring together the embeddings of the same plant across different modalities while pushing apart those of different plants.

The performance of this retrieval-focused model was evaluated using metrics such as Recall@k and Mean Average Precision (mAP) for all six possible retrieval directions (e.g., image-to-text, graph-to-image). To visualize the learned common embedding space, Uniform Manifold Approximation and Projection (UMAP) was applied to reduce the embeddings to two dimensions. A final figure was generated to illustrate the clustering of the different modalities in this space, and a histogram quantified the pairwise Euclidean distances between them.
