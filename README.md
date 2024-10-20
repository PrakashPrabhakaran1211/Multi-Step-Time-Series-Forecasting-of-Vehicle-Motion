# Multi-step Time Series Forecasting of Vehicle Motion

This repository contains code and resources for the project on multi-step time series forecasting of vehicle motion using both physics-based and data-driven techniques. The goal of this study is to predict the future motion of autonomous vehicles using two distinct methodologies: a physics-based approach and data-driven models.

## Dataset
The dataset used for this project is acquired from the **inD dataset**, which can be requested for non-commercial use from the [official website](https://levelxdata.com/ind-dataset/).

## Project Objective
The objective of this project is to explore physics-based modeling and data-driven techniques for predicting the motion of autonomous vehicles. The two approaches considered are:
- **Physics-based approach**: Utilizing the bicycle model and constant acceleration assumptions.
- **Data-driven techniques**: Implementing Multi-Layer Perceptron (MLP), Gated Recurrent Unit (GRU), and Long Short-Term Memory (LSTM) architectures.

These models are evaluated using a set of comprehensive metrics:
- Average Heading Error
- Final Displacement Error
- Average Absolute Heading Error

## Repository Structure
The repository is organized as follows:
1. **data_processing**: Contains Python files for reading, preprocessing, and preparation of the data.
2. **prediction_models**: Contains implementations of the Bicycle, constant acceleration, and data-driven (LSTM, MLP, GRU) models.
3. **evaluation**: Python files for evaluating the models based on the defined metrics.
4. **trained_models**: Stores the trained models in `.h5` format.
5. **loss_visualisation**: Contains plots of loss comparisons for the models (latest run of each model).
6. **results**: An Excel file containing the predicted and true values of the selected features in both the physics-based and data-driven models.
7. **main.ipynb**: The main Jupyter notebook for selecting models, training, and testing data IDs, and for executing all the above-mentioned Python modules.

## How to Execute
To run the models and make predictions, follow these steps:
1. Open the **main.ipynb** file.
2. Choose the required model using **Section 1.3 (Model Selection)** in the notebook.
3. For physics-based models:
   - The testing ID can be selected in **Section 1.0 (Data Reading)**.
   - The same testing ID can be used for the selection of training data for Data-Driven Models.
4. For data-driven models:
   - The testing ID can be chosen using **Section 3.1 (Generate Test Dataset)** in the notebook.
5. Once the models and recording IDs are selected, execute the notebook to run the training and evaluation processes.
6. The trained model, loss comparison plots, and results will be stored in their respective folders.

## Prerequisites
To get started with the project, make sure you have the following installed:
- Python 3.7 or higher
- Required libraries mentioned in `requirements.txt` (if available)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-step-time-series-forecasting.git
