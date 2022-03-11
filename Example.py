"""
Example.py

Provides usage examples for how to work with this software

Bryce Harrington
03/11/2022
"""
from TrafficLightDetector import TrafficLightDetector
from TrafficLightTrainer import TrafficLightTrainer
from Data.DataHandler import DataHandler


def main():
    # setup a data handler object so we have a uniform data source
    data_handler = DataHandler()
    # create a new traffic light detection model
    tlt = TrafficLightTrainer(data_handler=data_handler, model_name="test_model.h5")
    # create an instance of the traffic light detector so we can inference with the model
    tld = TrafficLightDetector("test_model.h5")


if __name__ == "__main__":
    main()