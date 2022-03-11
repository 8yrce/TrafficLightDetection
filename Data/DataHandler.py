"""
DataHandler

Takes in our raw data / labels into an ML friendly format. Handles pulling statistics from data as well

Bryce Harrington
03/11/2022

DataHandler - wrangles our data into a friendly and easily controllable package
        class DataHandler
            __init__(target_size = (300,300), ttv_split = [.7, .2, .1], data_path="Data/")
                self.images, self.labels = [],[]
            __load_data()
                Take our data into an ML friendly format
            get_classes()
                Returns the classes present in our dataset
"""
# system imports
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os

# project imports


class DataHandler:
    def __init__(self, target_size: tuple = (300, 300, 3), split=[.7, .2, .1]):
        self.target_size, self.split = target_size, split
        self.images, self.labels, self.partitions = [], [], []

        self.__populate_data_handler()

    def __populate_data_handler(self):
        """
        Take our data into an ML friendly format from our Data/Images and Data/Labels dir
        :sets: self.images, self.labels
        """
        def __load_data():
            """
            Load in the raw image / label data into our class
            """
            for file in os.listdir(__file__.replace(__file__.split("/")[-1], "") + "Images"):
                # load in the image, make sure its valid
                # print(file)
                # TODO replace with cv2 so we can auto scale
                image = load_img(__file__.replace(__file__.split("/")[-1], "") + "Images/" + file, self.target_size)
                # standardize / normalize image
                image = np.array(image) / 255.0

                # find label
                label = open(__file__.replace(__file__.split("/")[-1], "") + "Labels/" + file.replace(file.split(".")[-1], "txt"))

                # if they both check out add them to self.images/labels
                self.images.append(image)
                labels = []
                for line in label.readlines():
                    label = []
                    for num in line.split(" "):
                        label.append(float(num))
                    labels.append(np.array(label))
                self.labels.append(np.array(labels))
            # print(self.labels, self.images)

        def __partition_data():
            """
            Partition our data into test, train and val partitions
            """
            #calculate split
            splits = [int(split*len(self.images)) for split in self.split]
            print("Split: (train, test, val) = ", splits)

            for split in splits:
                part_images, part_labels = [], []
                for i in range(split):
                    part_images.append(self.images[i])
                    part_labels.append(self.labels[i])
                self.partitions.append([np.array(part_images), np.array(part_labels)])
            # print(self.partitions)

        __load_data()
        __partition_data()

    def get_subset(self, subset: str):
        """
        Returns a subset of the partioned data ( test, train, val )
        :param subset: subset of data you want to recieve ( string 'test,train,val' )
        :return: subset - the subset of data requested ( np.array )
        """
        if "train" in subset.lower():
            return self.partitions[0]
        elif "test" in subset.lower():
            return self.partitions[1]
        elif "val" in subset.lower():
            return self.partitions[2]
        else:
            print("[ERROR]: Unknown subset '{}'. Available options are train, test and val".format(subset))
            return None

    def get_classes(self):
        """
        Get the classes present in our data
        :return: self.labels classes
        """
        classes = []
        for labels in self.labels:
            for label in labels:
                if label[0] not in classes:
                    classes.append(label[0])
        return classes


if __name__ == "__main__":
    dh = DataHandler()
    print(dh.get_classes())
