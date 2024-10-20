import numpy as np
import math


#################################### Main Model Function ##################################################

def my_bicycle_model(test_data, pred_horizon, samp_time, frame_range):
    prediction = list()
    for frame_idx in range(frame_range):
        start_frame = frame_idx
        end_frame = frame_idx + pred_horizon
        pred_data = test_data[test_data['frame'] <= frame_idx]

        track_ids = test_data.loc[test_data['frame'] == frame_idx]['trackId']

        for track_id_idx in track_ids:
            prediction.append(my_prediction(pred_data, frame_idx, track_id_idx, pred_horizon, samp_time))

    return prediction


#################################### Predict Function ##################################################

def my_prediction(predData, currFrame, trackID, predHorizon, samplingTime):

    # heading prediction
    headingInit = list(predData.loc[(predData['frame'] == currFrame) &
                                    (predData['trackId'] == trackID), 'heading'])


    # xCenter prediction
    xVelInit = float(predData.loc[(predData['frame'] == currFrame) &
                                  (predData['trackId'] == trackID), 'xVelocity'])
    xCenterInit = float(predData.loc[(predData['frame'] == currFrame) &
                                     (predData['trackId'] == trackID), 'xCenter'])

    # yCenter prediction
    yVelInit = float(predData.loc[(predData['frame'] == currFrame) &
                                  (predData['trackId'] == trackID), 'yVelocity'])
    yCenterInit = float(predData.loc[(predData['frame'] == currFrame) &
                                     (predData['trackId'] == trackID), 'yCenter'])

    L = float(predData.loc[(predData['frame'] == currFrame) &
                                     (predData['trackId'] == trackID), 'length'])


    try:
        xHeadVel = xVelInit / L
    except ZeroDivisionError:
        xHeadVel = xVelInit # Default to heading if division by zero occurs
    try:
        beta =  np.arctan(yVelInit / xVelInit)
    except ZeroDivisionError:
        beta = 1  # Default to heading if division by zero occurs

    beta = np.ones(predHorizon) * beta
    x0 = np.ones(predHorizon) * xCenterInit
    a = np.arange(1, predHorizon + 1, 1)
    b = np.identity(predHorizon) * xVelInit * np.cos(headingInit +  beta) * samplingTime
    xCenter = list(x0 + np.matmul(a, b))



    y0 = np.ones(predHorizon) * yCenterInit
    b = np.identity(predHorizon) * yVelInit *np.sin(headingInit +  beta ) * samplingTime
    yCenter = list(y0 + np.matmul(a, b))

    z0 = np.ones(predHorizon) * headingInit
    b = (xHeadVel * np.sin(beta) * samplingTime)
    heading = list(z0 + np.matmul(a,b))
    prediction = xCenter + yCenter + heading
    #print ((prediction))
    return prediction

