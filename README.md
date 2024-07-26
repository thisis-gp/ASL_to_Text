# Holistic Sign Language Detection

This project uses OpenCV, MediaPipe, and TensorFlow to detect hand and facial landmarks, and predicts sign language gestures using a trained model.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model and Data](#model-and-data)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction

The goal of this project is to implement a real-time sign language detection system using computer vision and machine learning techniques. It utilizes the MediaPipe library to detect facial and hand landmarks and a pre-trained TensorFlow Lite model to classify the gestures.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/thisis-gp/ASL_to_Text
   cd ASL_to_Text
   ```
   
2. Create a virtual environment (optional but recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```

3. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. Ensure you have a webcam connected to your computer.

2. Run the detection script:
  ```bash
  python holistic_cam.py
  ```

3. The application will open a window displaying the webcam feed with detected landmarks and predicted signs.

4. Press q to exit the application.

## Model and Data
1. The project uses a TensorFlow Lite model (model.tflite) for prediction, which should be placed in the project directory. The model was trained on a dataset of sign language gestures.
2. train.csv: Contains the training data labels.
3. sample.parquet: A sample file for testing data processing.

## Contributing
Contributions are welcome! If you'd like to improve this project, please fork the repository, create a new branch, and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/NewFeature)
3. Commit your Changes (git commit -m 'Add some NewFeature')
4. Push to the Branch (git push origin feature/NewFeature)
5. Open a Pull Request

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

Feel free to customize this template further according to your project's specifics,  such as the repository URL, detailed descriptions, or additional setup instructions.
