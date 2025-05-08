# Digital Pheno-Parenting: A Deep Learning Model for Plant Phenotypes and Care

## 📌 Project Overview
Digital Pheno-Parenting is an advanced deep learning framework designed to identify plant species and detect leaf diseases from RGB images. It integrates three pretrained convolutional neural networks into an ensemble architecture, LeafNet, to achieve high classification accuracy. In addition to disease prediction, the system provides automated plant care recommendations. The solution is deployed through a user-friendly Streamlit interface.

## 🌿 Objectives
- Accurately classify plant species and associated leaf diseases.
- Leverage model ensembling (ResNet, EfficientNet, MobileNet) for improved performance.
- Offer actionable plant care suggestions based on the classification.
- Deploy an accessible interface for real-time predictions.

## 📁 Dataset
The dataset comprises approximately 87,000 labeled RGB images representing 12 plant species and 38 disease/healthy classes. It is split 80% for training and 20% for testing, with balanced class distribution to ensure model fairness.

## 🧠 Model Architecture
LeafNet combines the feature extractors of ResNet18, EfficientNet-B7, and MobileNet V3-Large. Their outputs are concatenated and passed through a shared fully connected layer for final classification. All base models are fine-tuned, and redundant heads are removed to streamline the ensemble.

## ⚙️ Training Details
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Scheduler:** ReduceLROnPlateau  
- **Regularization:** Batch Normalization, Dropout, Early Stopping  

## 📊 Results
- **ResNet Accuracy:** 98.63%  
- **EfficientNet Accuracy:** 93.92%  
- **MobileNet Accuracy:** 98.05%  
- **LeafNet Final Accuracy:** 99.29%  

## 🧩 Care Suggestion Module
Upon disease classification, the system provides concise, species-specific care guidance—covering watering frequency, fertilizer recommendations, and pest management—to assist farmers and plant owners.

## 🚀 Deployment
The model is deployed via Streamlit, offering:
- An upload portal for leaf images  
- Real-time disease prediction  
- Integrated care suggestions based on diagnosis  

## 🔧 Technologies Used
- Python 3.11  
- PyTorch, Torchvision  
- Scikit-learn, Pandas  
- OpenCV, Matplotlib  
- Streamlit  

## 📄 License
This project is licensed under the MIT License.

> ⚠️ **Note:** The trained model file (`LeafNet.pth`) and supporting resources (`class_mapping.json`, `suggestions.json`) are not included in this repository due to environment constraints. 
To use the Streamlit app, first train the model using the provided notebook and export these files accordingly.

## 📥 Dataset Access

This project uses the publicly available [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle.

To use this dataset:

1. Visit the dataset page: [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Download the ZIP file manually (requires a Kaggle account)
3. Extract the contents into the `data/` directory inside this project before training or testing the model

> ⚠️ Due to its size (~2.7GB), the dataset is not included in this repository.

## 👥 Project Acknowledgment

**Team Lead:** B. Bhuvan Kumar  
**Mentor:** Prof. K. Mahammad

This project was developed as part of a collaborative research effort focused on deep learning applications in plant phenotyping and disease classification.