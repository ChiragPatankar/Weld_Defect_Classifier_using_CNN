# Weld Defect Classification

## Overview
This project is a deep learning-based system for classifying weld defects using convolutional neural networks (CNNs). The system processes weld images and predicts whether a weld is good or defective, categorizing defects into six different classes.

## Features
- **Trainable Model**: Uses PyTorch to train a CNN on labeled weld images.
- **Testing Module**: Evaluates model accuracy using test images.
- **Web Interface**: A Streamlit-powered UI for real-time image classification.
- **Binary and Multi-Class Classification**: Can classify welds as `Good` or `Defective`, and further categorize defects into specific types.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. You also need the following dependencies:
```sh
pip install torch torchvision streamlit numpy pandas matplotlib pillow
```

## Usage

### Training the Model
To train the model, run:
```sh
python train.py
```
This will load training images, process them, and train a CNN model. The trained model is saved as `modelv5.pt`.

### Testing the Model
After training, evaluate the model using:
```sh
python test.py
```
This script loads test images and prints evaluation metrics such as accuracy.

### Running the Web App
To launch the Streamlit UI for classifying weld images, run:
```sh
streamlit run app.py
```
This will start a web-based interface where users can upload images and receive predictions.

## Project Structure
```
├── data/                  # Directory for training and test datasets
├── train.py               # Training script
├── test.py                # Testing script
├── app.py                 # Streamlit UI for classification
├── modelv5.pt             # Trained model file (generated after training)
└── README.md              # Project documentation
```

## Model Details
- CNN architecture with multiple convolutional, batch normalization, dropout, and ReLU layers.
- Uses cross-entropy loss and Adam optimizer for training.
- Outputs a classification prediction with probability scores.

## Defect Classes
1. **Good Weld**
2. **Burn Through**
3. **Contamination**
4. **Lack of Fusion**
5. **Misalignment**
6. **Lack of Penetration**

## Example Output
Below is a sample output from the web application:

![Sample Output](images/Screenshot 2025-03-16 015948.png)

## Future Enhancements
- Improve accuracy by fine-tuning the model.
- Extend dataset with more labeled weld images.
- Deploy as a cloud-based service for industrial use.

## License
This project is licensed under the MIT License.

