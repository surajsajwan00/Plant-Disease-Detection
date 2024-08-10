# Automated Plant Disease Detection for Precision Agriculture

## Overview

This project aims to develop an automated system for detecting and classifying plant diseases using image analysis techniques. Leveraging deep learning, specifically Convolutional Neural Networks (CNNs), this system helps farmers and agricultural professionals in early disease detection, enabling timely interventions to mitigate crop losses and enhance agricultural productivity.

## Introduction

The primary objective of this project is to develop an automated system capable of accurately detecting and classifying plant diseases using image analysis techniques. By leveraging deep learning, specifically Convolutional Neural Networks (CNNs), the system aims to assist farmers and agricultural professionals in early disease detection, enabling timely interventions to mitigate crop losses and enhance agricultural productivity.

## Technologies Used

- **Deep Learning**: VGG16 architecture for image classification
- **Python**: The core programming language used for development
- **Libraries**: NumPy, Pandas, OpenCV, TensorFlow/Keras
- **Streamlit**: A Python library for creating the user interface
- **Data**: Kaggle "Plant Leaves Disease Dataset" with over 70,000 images

## System Architecture

The system comprises the following components:

1. **Data Acquisition and Preprocessing Module**: Collects and preprocesses image data.
2. **Model Development and Training Module**: Builds and trains the VGG16 model.
3. **Disease Detection and Classification Module**: Predicts diseases from input images.
4. **User Interface Module**: Streamlit-based UI for interacting with the model.
5. **Deployment Module**: (Future integration) For deploying the system to a production environment.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Virtual environment tool (e.g., venv, conda)
- An internet connection to download dependencies

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/surajsajwan00/Plant-Disease-Detection.git
   cd Plant-Disease-Detection

2. **Install the Required Dependencies**
   To install the required dependencies, use the following command:
   ```bash
   streamlit run app.py

4. **Running the Streamlit Application**
   Start the Streamlit server with the following command:
      ```bash
   streamlit run app.py

## Future Development
- **Mobile Application**: Develop a mobile app for real-time disease detection in the field.
- **IoT Integration**: Integrate IoT devices for environmental monitoring.
- **Expert Analysis**: Include expert analysis and recommendations for treatment.

## Author
Suraj singh Sajwan-[GitHub](https://github.com/surajsajwan00)

## Acknowledgments
- A big thanks to Kaggle for providing the "Plant Leaves Disease Dataset."
- Inspiration for the project and guidance from online communities and mentors.

