
## Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation with modification to the model characteristics & experiment with a standard medical dataset named as KVASIR-Seg 
##  Overview

This project addresses the challenge of medical image classification using self-supervised learning techniques. We adapt and apply the approach inspired by the CVPR 2023 paper "Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation" to the domain of gastrointestinal endoscopy images.

Key features:
- Implementation of self-supervised learning methods for medical image classification
- Application to the Kvasir dataset, a benchmark for gastrointestinal tract image classification
- Evaluation on multi-class classification tasks
- Achieved state-of-the-art performance with 92.57% average classification accuracy

The repository includes code for training models with self-supervised techniques, as well as fine-tuning and evaluation scripts specifically tailored for the Kvasir dataset.

## Performance Highlights

The method demonstrates significant improvements in classification accuracy on the Kvasir dataset:

- Average Classification Accuracy: 92.57%

This performance represents a substantial advancement in automated analysis of gastrointestinal endoscopy images, potentially aiding in more accurate and efficient medical diagnoses.

## Dataset

The Kvasir dataset consists of gastrointestinal tract images collected during endoscopic examinations, including multiple classes such as ulcerative colitis, polyps, esophagitis, and normal findings. Our model's high performance on this diverse and challenging dataset underscores its potential for real-world clinical applications.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`Ubuntu 18.04`
`Python 3.10`
`CUDA 11.0`
`pyTorch 1.12.0`

## The package requirements can be installed via `environment.yml` , which includes
+ `pyTorch 1.12.0`
+ `torrchvision 0.13.0`
+ `cudatoolkit 11.3`
+ `pytorch-lightning 1.6.5`
+ `einops 0.6.0`
``conda env create --name pytorch1.12 --file environment.yml -p YOURCONDADIR/envs/pytorch1.12
conda activate pytorch1.12``
-
Make sure to replace `YOURCONDADIR` in the installation path with your conda dir, e.g., `~/anaconda3`


## Datasets
+ KVASIR 
#### The Kvasir-SEG dataset (size 46.2 MB) contains 1000 polyp images and their corresponding ground truth from the Kvasir Dataset v2. The resolution of the images contained in Kvasir-SEG varies from 332x487 to 1920x1072 pixels. The images and its corresponding masks are stored in two separate folders with the same filename. The image files are encoded using JPEG compression, and online browsing is facilitated. The open-access dataset can be easily downloaded for research and educational purposes.

#### The bounding box (coordinate points) for the corresponding images are stored in a JSON file. This dataset is designed to push the state of the art solution for the polyp detection task. 

### Ground Truth Extraction

###### We uploaded the entire Kvasir polyp class to Labelbox and created all the segmentations using this application. The Labelbox is a tool used for labeling the region of interest (ROI) in image frames, i.e., the polyp regions for our case. We manually annotated and labeled all of the 1000 images with the help of medical experts. After annotation, we exported the files to generate masks for each annotation. The exported JSON file contained all the information about the image and the coordinate points for generating the mask. To create a mask, we used ROI coordinates to draw contours on an empty black image and fill the contours with white color. The generated masks are a 1-bit color depth images. The pixels depicting polyp tissue, the region of interest, are represented by the foreground (white mask), while the background (in black) does not contain positive pixels. Some of the original images contain the image of the endoscope position marking probe, ScopeGuide TM, Olympus Tokyo Japan, located in one of the bottom corners, seen as a small green box. As this information is superfluous for the segmentation task, we have replaced these with black boxes in the Kvasir-SEG dataset.

+ Trainining with pixel-level supervision
` python main.py --datapath YOUR_DATASET_DIR \
               --benchmark kvasir-seg \
               --logpath YOUR_DIR_TO_SAVE_CKPT \
               --fold {0, 1, 2, 3} \
               --sup mask `
+ Training with image-level supervision
` python main.py --datapath YOUR_DATASET_DIR \
               --benchmark {pascal, coco} \
               --logpath YOUR_DIR_TO_SAVE_CKPT \
               --fold {0, 1, 2, 3} \
               --sup pseudo `
## Authors

- [@Dahyun-Kang](https://github.com/dahyun-kang/cst/)

 author={Kang, Dahyun and Koniusz, Piotr and Cho, Minsu and Murray, Naila}
 ` year: 2023 `