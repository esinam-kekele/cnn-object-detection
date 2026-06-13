# 🚗 CNN Object Detection: License Plate Detection

<p align="center">
  <a href="https://colab.research.google.com/github/kekele-star/number-plate-detection/blob/main/train-set.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

---

## 📖 Overview

This project introduces the fundamentals of **Object Detection** and demonstrates how to fine-tune a pre-trained deep learning model for **license plate detection**.

Object detection is one of the most important tasks in computer vision, enabling machines to identify and locate multiple objects within an image. This project combines both theory and hands-on implementation using modern deep learning techniques.

---

## 🎯 Learning Objectives

By the end of this project, you will understand:

- Object Detection fundamentals
- Key components of detection architectures
- Popular object detection models
- Dataset formats and annotation techniques
- Model evaluation metrics
- Inference using pre-trained models
- Fine-tuning on custom datasets
- License plate detection implementation

---

## 📑 Table of Contents

- [Overview of Object Detection](#-overview-of-object-detection)
- [Key Components of Object Detection Models](#-key-components-of-object-detection-models)
- [Popular Object Detection Models](#-popular-object-detection-models)
- [Tools and Libraries](#-tools-and-libraries)
- [Data Preparation](#-data-preparation)
  - [Data Formats](#data-formats)
  - [Data Annotation](#data-annotation)
- [Inference with Pre-trained Models](#-inference-with-pre-trained-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Fine-Tuning on License Plate Dataset](#-fine-tuning-on-license-plate-dataset)

---

# 🔍 Overview of Object Detection

Object Detection is a computer vision task that identifies **what objects are present in an image** and **where they are located**.

Unlike image classification, which assigns a single label to an image, object detection predicts:

- Object class labels
- Bounding box coordinates
- Confidence scores

### Common Applications

- 🚘 Autonomous Vehicles
- 🎥 Surveillance Systems
- 📦 Inventory Management
- 🏭 Industrial Inspection
- 🔍 Visual Search Engines

<p align="center">
  <img src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/68/ae/object-detection-figure-1.component.complex-narrative-xl.ts=1713813010967.jpg/content/adobe-cms/us/en/topics/object-detection/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative/items/content_group_745532756/image" width="700">
</p>

---

# 🧠 Key Components of Object Detection Models

## 1. Feature Extraction

Feature extraction is the process of learning meaningful visual patterns from images.

### Common CNN Backbones

- ResNet
- VGG
- MobileNet
- EfficientNet

Pre-trained models leverage knowledge learned from large datasets such as ImageNet, reducing training time and improving performance.

---

## 2. Region Proposal

Region proposal methods identify potential object locations before classification.

### Popular Approaches

- Selective Search
- Region Proposal Networks (RPNs)

RPNs are commonly used in Faster R-CNN architectures.

---

## 3. Classification & Localization

### Object Classification

Assigns a class label to detected objects.

### Bounding Box Regression

Predicts object coordinates:

```text
[x, y, width, height]
```

---

## 4. Anchor Boxes

Anchor boxes are predefined bounding boxes of varying sizes and aspect ratios used to detect objects at different scales.

Commonly used in:

- YOLO
- SSD
- Faster R-CNN

---

## 5. Loss Functions

### Classification Loss

Measures class prediction accuracy.

### Localization Loss

Measures bounding box prediction accuracy.

### Total Loss

Combines classification and localization losses into a single optimization objective.

---

## 6. Post-Processing

### Non-Maximum Suppression (NMS)

NMS removes duplicate detections and retains the most confident bounding boxes.

---

# 🚀 Popular Object Detection Models

## R-CNN (Regions with CNN Features)

### Characteristics

- Uses Selective Search for region proposals
- CNN extracts features
- SVM performs classification

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*yJBiHhK8t_zTQKBqlZKlWQ.png" width="650">
</p>

---

## Fast R-CNN

### Improvements

- Shared convolution computations
- Faster training
- ROI Pooling

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*7haXXOJjZdibibU0c4eNgw.png" width="650">
</p>

---

## Faster R-CNN

### Improvements

- Introduces Region Proposal Networks (RPNs)
- End-to-end training
- Higher detection accuracy

<p align="center">
  <img src="https://www.researchgate.net/publication/326668850/figure/fig1/AS:653294032666626@1532768843320/Faster-R-CNN-architecture-Top-left-box-represents-the-base-network-box-on-the-right.png" width="650">
</p>

---

## YOLO (You Only Look Once)

### Strengths

- Real-time detection
- High speed
- Single-stage architecture

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/1512812.png" width="650">
</p>

---

## SSD (Single Shot MultiBox Detector)

### Strengths

- Single forward pass
- Fast inference
- Multi-scale feature maps

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*La_I2VXlENAJ9r0Wpf_vMg.jpeg" width="650">
</p>

---

# 🛠️ Tools and Libraries

| Library | Purpose |
|----------|----------|
| OpenCV | Image processing and computer vision |
| TensorFlow | Deep learning framework |
| PyTorch | Deep learning and research |
| Detectron2 | Object detection and segmentation |

---

# 📂 Data Preparation

## Data Formats

Object detection datasets are stored in several annotation formats.

### 1. COCO Format

The most widely used object detection format.

Includes:

- Images
- Annotations
- Categories
- Segmentation masks
- Keypoints

#### Bounding Box Format

```text
[x, y, width, height]
```

---

### 2. Pascal VOC Format

XML-based annotation format used in the PASCAL Visual Object Classes Challenge.

#### Bounding Box Format

```text
[xmin, ymin, xmax, ymax]
```

---

### 3. YOLO Format

Text-based annotation format used by YOLO models.

#### Structure

```text
<class_id> <x_center> <y_center> <width> <height>
```

#### Example

```text
0 0.4921875 0.3958333 0.25390625 0.375
1 0.244140625 0.4708333 0.140625 0.175
```

---

### 4. Custom Datasets

Custom datasets often require additional preprocessing and annotation conversion before training.

---

## Data Annotation

Data annotation involves labeling images with:

- Bounding boxes
- Class labels

### Popular Annotation Tools

| Tool | Description |
|--------|-------------|
| LabelImg | Simple image annotation tool |
| CVAT | Advanced annotation platform |
| Roboflow | Annotation and dataset management |
| VIA | Lightweight browser-based annotation tool |

<p align="center">
  <img src="https://blog.roboflow.com/content/images/2020/12/labeling.small-1.gif" width="650">
</p>

---

# 🤖 Inference with Pre-trained Models

In this section, we use a pre-trained object detection model to perform inference on unseen images.

Key steps include:

1. Loading a trained model
2. Preprocessing input images
3. Running predictions
4. Visualizing detected objects

---

# 📊 Evaluation Metrics

Object detection performance is typically evaluated using:

### Intersection over Union (IoU)

Measures overlap between predicted and ground-truth bounding boxes.

### Precision

Measures the percentage of correct positive predictions.

### Recall

Measures the percentage of actual objects detected.

### Mean Average Precision (mAP)

The most widely used object detection metric.

---

# 🎯 Fine-Tuning on License Plate Dataset

In the final section, we fine-tune a pre-trained object detection model on a custom License Plate Dataset.

The workflow includes:

- Dataset preparation
- Annotation processing
- Model training
- Validation
- Performance evaluation
- Inference on test images

---

## 📌 Key Takeaways

✅ Understanding Object Detection fundamentals

✅ Exploring modern detection architectures

✅ Working with common annotation formats

✅ Evaluating detection performance

✅ Fine-tuning models on custom datasets

✅ Building a License Plate Detection system

---
