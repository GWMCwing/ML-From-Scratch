from typing import Tuple
import numpy as np


class NaiveBayesClassifier:

    def __init__(self, featureSize: int, labelSize: int, discrete: bool = True):
        """ constructor for Naive Bayes Classifier
        Args:
            featureSize (int): number of features
            labelSize (int): number of classifier output
            discrete (bool, optional): Use discrete method. Defaults to True.
        """

        self.featureSize: int = featureSize
        self.labelSize: int = labelSize
        self.discrete: bool = discrete
        self.train = self.__trainDiscrete if discrete else self.__trainContinuous
        #
        self.prior: np.ndarray = np.zeros(labelSize)
        self.likelihood: np.ndarray = np.zeros((labelSize, featureSize))
        self.posterior: np.ndarray = np.zeros(labelSize)
        #
        self.totalTrainLabelsCount = np.zeros((labelSize))

    def verifyFeature(self, features: np.ndarray) -> bool:
        # TODO
        return True

    def verifyLabel(self, labels: np.ndarray) -> bool:
        if (labels.argmax() + 1 > self.labelSize):
            return False
        return True

    def __estimateClassPrior(self, labels: np.ndarray, trainingSize: int) -> Tuple[np.ndarray, np.ndarray]:
        unique, counts = np.unique(labels, return_counts=True)
        class_prior = np.zeros(self.labelSize)
        class_prior[unique] = counts
        return (class_prior + 1)/(trainingSize + self.labelSize), counts

    def __estimateLikelihoods(self, features: np.ndarray):
        featureMax = features.argmax(1)+1
        likelihoods = np.zeros((self.featureSize, self.labelSize, featureMax))

        pass

    def __trainDiscrete(self, features: np.ndarray, labels: np.ndarray):

        pass

    def __trainContinuous(self, features: np.ndarray, labels: np.ndarray):
        pass

    def predict(self, features) -> np.ndarray:
        pass
