import numpy as np
import pandas as pd
from os.path import exists


class evaluationMatrixCa:
    def __init__(self, filename, predictHorizon):
        self.pred_filename = filename
        self.predict_horizon = predictHorizon
        self.pred_results = ""
        self.file_exists_check = False
        self.eval_data_exists = False
        self.fde_val = 0  # Final displacement error
        self.ade_val = 0  # Average displacement error

        self.x_val = 0
        self.x_hat = 0
        self.x_error = 0  # x - x_hat
        self.x_error_sqre = 0  # (x - x_hat)^2

        self.y_val = 0
        self.y_hat = 0
        self.y_error = 0  # y - y_hat
        self.y_error_sqre = 0  # (y - y_hat)^2

        self.error_sqre_sum = 0  # (x - x_hat)^2 + (y - y_hat)^2
        self.error_sum_sqrt = 0  # sqrt((x - x_hat)^2 + (y - y_hat)^2)



    def read_prediction_file(self):
        self.file_exists_check = exists(self.pred_filename)
        if self.file_exists_check:
            self.pred_results = pd.read_excel(self.pred_filename, engine='openpyxl')
            self.eval_data_exists = True
        else:
            print("Evaluation file does not exist")

    def calculate_fde(self):
        # final displacement error
        self.x_hat = self.pred_results.iloc[:, 3 + self.predict_horizon - 1].to_numpy()
        self.y_hat = self.pred_results.iloc[:, 3 + 2 * self.predict_horizon - 1].to_numpy()

        self.x_val = self.pred_results.iloc[:, 3 + 3 * self.predict_horizon - 1].to_numpy()
        self.y_val = self.pred_results.iloc[:, 3 + 4 * self.predict_horizon - 1].to_numpy()

        self.x_error = np.subtract(self.x_val, self.x_hat)
        self.y_error = np.subtract(self.y_val, self.y_hat)
        self.x_error_sqre = np.power(self.x_error, 2)
        self.y_error_sqre = np.power(self.y_error, 2)
        self.error_sqre_sum = np.add(self.x_error_sqre, self.y_error_sqre)
        self.error_sum_sqrt = np.sqrt(self.error_sqre_sum)

        self.error_sum_sqrt = np.nan_to_num(self.error_sum_sqrt)

        if self.error_sum_sqrt.size:
            self.fde_val = (1 / self.error_sum_sqrt.size) * np.sum(self.error_sum_sqrt)
        else:
            self.fde_val = 0

    def calculate_ade(self):
        # average displacement error
        self.x_hat = self.pred_results.iloc[:, 3:3 + self.predict_horizon].to_numpy()
        self.y_hat = self.pred_results.iloc[:, 3 + self.predict_horizon:3 + 2 * self.predict_horizon].to_numpy()

        self.x_val = self.pred_results.iloc[:, 3 + 2 * self.predict_horizon:3 + 3 * self.predict_horizon].to_numpy()
        self.y_val = self.pred_results.iloc[:, 3 + 3 * self.predict_horizon:3 + 4 * self.predict_horizon].to_numpy()

        self.x_error = np.subtract(self.x_val, self.x_hat)
        self.y_error = np.subtract(self.y_val, self.y_hat)
        self.x_error_sqre = np.power(self.x_error, 2)
        self.y_error_sqre = np.power(self.y_error, 2)
        self.error_sqre_sum = np.add(self.x_error_sqre, self.y_error_sqre)
        self.error_sum_sqrt = np.sqrt(self.error_sqre_sum)

        self.error_sum_sqrt = np.nan_to_num(self.error_sum_sqrt)

        if self.error_sum_sqrt.size:
            self.ade_val = (1 / self.error_sum_sqrt.size) * np.sum(self.error_sum_sqrt)
        else:
            self.ade_val = 0




    def get_result(self):
        self.read_prediction_file()
        # No evaluation if data does not exists
        if self.eval_data_exists:
            self.calculate_fde()
            self.calculate_ade()
        self.print_fde_ade()
        return self.ade_val, self.fde_val

    def print_fde_ade(self):
        print('The average displacement error is ' + str(round(self.ade_val, 3)) + ' m')
        print('The average final displacement error is ' + str(round(self.fde_val, 3)) + ' m')
