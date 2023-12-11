
# README for P1 Project - Flight Delay Prediction

## Project Description
This project is part of our P1 project at AAU. The aim is to predict flight delays using machine learning models. We utilize a dataset from Kaggle, which includes flight data from 2018 to 2022.

## Code Description
The code consists of several parts:

1. **Data Loading and Preprocessing:** We load flight data and prepare it for analysis by performing data cleaning and feature engineering.
   
2. **Data Splitting and Balancing:** Data is split into training and test sets. We also use SMOTENC to balance the data.

3. **Model Training and Evaluation:** Various classification models, including Random Forest, Decision Tree, Naive Bayes, and K-Nearest Neighbors, are trained and evaluated on the dataset.

4. **Model Saving and Performance Logging:** The trained models are saved, and their performance is logged for further analysis.

## Setup

### Requirements
- Python 3.8 or newer
- All necessary Python packages are listed in `requirements.txt`.

### Installation
1. Clone this repository.
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Downloading the Dataset
To use this code, you need to download the dataset from Kaggle:
[Flight Delay Dataset 2018-2022](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022/data). Place the data file in the same folder as the project code.

### Running the Program
To run the program, open a terminal in the project's root folder and execute:
```
python main.py
```

## Additional Information
- The `main()` function initializes and trains models on the specified data columns.
- The results of the models' performance are saved in separate files for each model.
- Data preprocessing and model training can take considerable time depending on your computer's performance.

## Contact
For questions or further information, please contact us via our project email.

---
**Note:** This README is a template and can be adapted to the specific requirements and development stage of the project.
