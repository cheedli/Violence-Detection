# Violence Detection System

## Overview

Welcome to our Violence Detection System! This solution is designed to analyze video content and identify instances of violence, such as kicking or shooting. Leveraging the power of OpenCV, Flask, and TensorFlow libraries, our system provides a robust and efficient way to detect violent activities in videos.
## Dataset
the dataset was taken from https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
## Features

- **Video Analysis**: Our system captures video elements to thoroughly analyze and detect violent actions within the content.

- **Flexibility**: By utilizing OpenCV, Flask, and TensorFlow, our solution is versatile and can be easily integrated into different applications and platforms.

- **Accuracy**: TensorFlow, a powerful machine learning library, enhances the accuracy of violence detection, making the system reliable and effective.
##
##
## Libraries Used

### 1. OpenCV

[OpenCV](https://opencv.org/) (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides a wide range of tools and functions for image and video analysis. In our system, OpenCV is instrumental in capturing video elements and extracting features necessary for violence detection.

![OpenCV Logo](https://github.com/opencv/opencv/raw/master/doc/opencv-logo2.png)
### 2. Flask

[Flask](https://flask.palletsprojects.com/) is a lightweight and flexible web framework for Python. We use Flask to create a user-friendly interface for our Violence Detection System. This allows users to interact with the system easily, providing a seamless experience for video analysis and result retrieval.

![Alternative Flask Logo](https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/flask-logo-icon.png)


### 3. TensorFlow

[TensorFlow](https://www.tensorflow.org/) is a popular open-source machine learning framework developed by Google. It is widely used for building and training machine learning models. In our solution, TensorFlow is utilized to enhance the accuracy of violence detection by implementing machine learning algorithms.

![TensorFlow Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png)
##
##
##
## Getting Started

To run the Violence Detection System locally, follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install opencv-python flask tensorflow

2. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/violence-detection.git

3. **Run the Application:**
    ```bash
    cd violence-detection
    python app.py

4.**Access the Interface:**
  Open your web browser and navigate to http://localhost:5000 to access the Violence Detection System.
  

##
##
##



# Violence Detection System

####
## Challenges Faced

### 1. Data Availability
- **Problem:** Initially struggled with a lack of training data for the violence detection model.
- **Solution:** After an exhaustive two-hour search, the team managed to secure a substantial dataset with 12GB size and 5 million images.

### 2. Training Setback
- **Problem:** The first attempt at training the model resulted in an unexpected accuracy of 0.16 after four hours of processing.
- **Solution:** Collaborated with a friend remotely, leveraging their powerful computer for training. Despite an unforeseen shutdown, the team persisted and reattempted with an alternative code, achieving an impressive 0.94 accuracy.

### 3. Deployment Challenges
- **Problem:** Faced difficulties in deploying the model, particularly in handling real-time feed data.
- **Solution:** Implemented a comprehensive solution to address deployment issues, ensuring the smooth processing of real-time video data.




##
##

## Lessons Learned

1. **Data Exploration is Crucial:** The initial struggle with data scarcity emphasizes the importance of thorough exploration to find a suitable dataset for training.

2. **Resilience Pays Off:** The setback during training could have demoralized the team, but your persistence and the decision to try again with a different approach led to success.

3. **Remote Collaboration Has Risks:** While remote collaboration is valuable, there are inherent risks, as seen when your friend's computer was unexpectedly turned off. It highlights the need for contingency plans.

4. **Iterate and Optimize:** The journey from a low accuracy of 0.16 to a high accuracy of 0.94 demonstrates the iterative nature of model development. Trying different approaches and optimizing code contributed to eventual success.

5. **Deployment is a Unique Challenge:** Transitioning from model development to deployment involves unique challenges. Identifying and resolving issues related to real-time data processing is a crucial aspect of the deployment phase.

6. **Team Resilience:** The challenges encountered during the hackathon served as a testament to the team's resilience. Overcoming setbacks and finding solutions collectively strengthened the team's bond and problem-solving skills.

## Model Used
The violence detection model in our system is based on MobileNet and LSTM (Long Short-Term Memory) architecture, commonly referred to as MobileLSTM. This architecture is particularly effective for real-time video analysis, providing both accuracy and efficiency.
## Getting Started

To run the Violence Detection System locally, follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install opencv-python flask tensorflow












## Contributors


- Amine Fezzani
- Taher Turki
- Chedly ghorbel
- Mohamed Ali Farhat






   
