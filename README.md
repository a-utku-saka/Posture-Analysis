# Posture Analyses for For Bodybuilding Exercises

By analysing the posture, it receives the user's posture information live from the camera. According to the exercise chosen by the user, the live image is processed with the artificial intelligence model and posture tracking is performed. If the user performs the movement correctly, it is confirmed with green lines and the repetition counter is increased. Thus, it is monitored whether the user performs the movement correctly or not, and the number of repetitions and sets of the movement is monitored.
## File Structure:

- `exercise_detection.py`: The defined angle ranges are adjusted according to the movement selected by the user..
- `pose_detection.py`: Pose tracking is done with live data received from the camera..
- `main.py`: The main file to run.

## Installation:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/Posture-Analysis.git
   cd Posture-Analysis
2. **Install the required dependencies:**

Ensure you have numpy, cv2, tensorflow, tensorflow_hub installed. 
If not, you can install it using pip:

  ```sh
  pip install numpy
  pip install cv2
  pip install tensorflow
  pip install tensorflow_hub
  ```
## Usage:
**Start the programme**
To train the Q-learning model, simply run:

 ```sh
python main.py
```

**Exercise Selection**

Select one of the exercises defined in the programme. 

**Customizing Training Parameters**
When the red lines on the screen turn green, the movement is done correctly. As a result, the repeat counter is increased by 1.

## Lisance 
This project is licensed under the MIT License.
