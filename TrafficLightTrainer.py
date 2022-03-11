"""
TrafficLightTrainer

Defines a network and fits it using a populated DataHandler into a trained TrafficLightDetection model

Bryce Harrington
03/11/2022

TrafficLightTrainer - trains a model to passed in traffic light data
    class TrafficLightTrainer
        __init__( model_name:string, dataHandler:DataHandler )
            self.model = None
            self.model_name = model_name
            self.dataHandler = dataHandler
        __create_network
            Specify the network architecture, compile
            input_layer.shape = dataHandler[0].shape
            output_layer.shape = len(dataHandler.get_classes())
        __train_model
            Train the spec'ed network so it's fit on the data
        evaluate_model( eval_data:np.array )
            Test models performance on a given eval dataset
"""
# system imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, Model, losses, optimizers, callbacks
import os

# project imports

from Data.DataHandler import DataHandler


class TrafficLightTrainer:
    def __init__(self, data_handler: DataHandler = None, model_name: str = "TrafficLightDetectionModel.h5"):
        self.data_handler, self.model_name = data_handler, model_name
        self.model = None

        # if model_name exists in model repo use it, if not make new with that name to store there
        if os.path.exists("Models/" + self.model_name):
            self.model = tf.keras.models.load_model("Models/" + self.model_name)
        # TODO handling for no dataHandler
        else:
            self.__create_network()
            self.__train_model()

    def __create_network(self):
        """
        Create the traffic light detection network
        :sets: self.model
        """
        # use datahandler to get input / output sizes
        base_network = tf.keras.applications.resnet.ResNet50(input_shape=self.data_handler.target_size,
                                                             weights="imagenet", include_top=False)
        base_network.trainable = False
        base_network.summary()

        flattened_base = base_network.output
        flattened_base = layers.Flatten()(flattened_base)

        # define bbox regression head
        bbox = layers.Dense(128, activation=activations.relu)(flattened_base)
        bbox = layers.Dropout(0.5)(bbox)
        bbox = layers.Dense(64, activation=activations.relu)(bbox)
        bbox = layers.Dropout(0.25)(bbox)
        bbox = layers.Dense(32, activation=activations.relu)(bbox)
        bbox = layers.Dense(4, activation=activations.sigmoid, name="bbox")(bbox)

        # define the classification head
        classification = layers.Dense(128, activation=activations.relu)(flattened_base)
        classification = layers.Dropout(0.5)(classification)
        classification = layers.Dense(64, activation=activations.relu)(classification)
        classification = layers.Dense(len(self.data_handler.get_classes()), activation=activations.softmax, name="classification")(classification)

        # define losses
        loss = {
            "bbox": losses.MeanSquaredError,
            "classification": losses.categorical_crossentropy
        }

        self.model = Model(inputs=base_network.inputs, outputs=(bbox, classification))
        self.model.compile(
            loss=loss,
            optimizer=optimizers.Adam(1e-4),
            metrics=["accuracy"]
        )

    def __train_model(self):
        """
        Train the generated network using the data handler to fit
        :sets: self.model
        """
        # generate network architecture
        self.__create_network()

        # define callbacks
        early_stopping = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        model_checkpoint = callbacks.ModelCheckpoint("Models/" + self.model_name)

        # pull train and val set from data handler
        train_data, train_labels = self.data_handler.get_subset("train")
        val_data, val_labels = self.data_handler.get_subset("val")

        self.model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                       callbacks=[early_stopping, model_checkpoint], epochs=1000)

    def evaluate_model(self, eval_data):
        """
        Evaluate the model using a set of eval data
        :param eval_data: data set for eval ( images / labels ) (np.array)
        """
        eval_output = ""
        if self.model:
            eval_output = self.model.eval(eval_data)
        return eval_output


if __name__ == "__main__":
    tlt = TrafficLightTrainer()
