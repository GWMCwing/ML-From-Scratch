from typing import Tuple
import numpy as np


class NaiveBayesClassifier:

    def __init__(self, featureSize: int, classSize: int, discrete: bool = True):
        """Constructor for Naive Bayes Classifier
        Args:
            featureSize (int): number of features
            labelSize (int): number of classifier output
            discrete (bool, optional): Use discrete method. Defaults to True.
        """

        self.featureSize: int = featureSize
        self.classSize: int = classSize
        self.discrete: bool = discrete
        self.train = self.__trainDiscrete if discrete else self.__trainContinuous
        #
        self.prior: np.ndarray = np.zeros(classSize)
        self.likelihood: np.ndarray = np.zeros((classSize, featureSize, 2))
        self.posterior: np.ndarray = np.zeros(classSize)
        #
        self.totalTrainingCount = 0
        self.totalTrainingClassCount = np.zeros(classSize)

    def verifyFeature(self, features: np.ndarray) -> bool:
        # TODO
        return True

    def verifyLabel(self, labels: np.ndarray) -> bool:
        if (labels.argmax() + 1 > self.classSize):
            return False
        return True

    def predict(self, features) -> np.ndarray:
        # TODO
        return self.posterior

    def __estimateClassPrior(self, labels: np.ndarray, trainingSize: int) -> Tuple[np.ndarray, np.ndarray]:
        unique, counts = np.unique(labels, return_counts=True)
        classCount = np.zeros(self.classSize)
        classCount[unique] = counts
        return (classCount + 1)/(trainingSize + self.classSize), classCount

    def __updatePrior(self, prior: np.ndarray, trainingSize: int):
        weightedAverage = self.totalTrainingCount / \
            (self.totalTrainingCount + self.classSize + trainingSize)
        self.prior *= weightedAverage
        self.prior += prior * (1 - weightedAverage)
        return self.prior

    def __estimateLikelihoods(self, features: np.ndarray):
        featureMax = features.argmax(1)+1
        likelihoods = np.zeros((self.featureSize, self.classSize, featureMax))
        # TODO
        return likelihoods

    def __updateLikelihoods(self, likelihoods: np.ndarray, trainingSize: int):
        weightedAverage = self.totalTrainingClassCount / \
            (self.totalTrainingClassCount + trainingSize)
        shapeDifference = likelihoods.shape[2] - self.likelihood.shape[2]
        if (shapeDifference > 0):
            # pad self.likelihood with 0 to match the new likelihoods
            self.likelihood = np.pad(
                self.likelihood, ((0, 0), (0, 0), (0, shapeDifference)))
        self.likelihood = self.likelihood * \
            weightedAverage + likelihoods * (1-weightedAverage)
        return self.likelihood

    def __trainDiscrete(self, features: np.ndarray, labels: np.ndarray):
        trainingSize = features.shape[0]
        prior, classCount = self.__estimateClassPrior(labels, labels.shape[0])
        likelihoods = self.__estimateLikelihoods(features)
        self.__updatePrior(prior, trainingSize)
        self.__updateLikelihoods(likelihoods, trainingSize)
        # TODO
        self.totalTrainingCount += features.shape[0]
        self.totalTrainingClassCount += classCount
        pass

    def __trainContinuous(self, features: np.ndarray, labels: np.ndarray):
        prior, classCount = self.__estimateClassPrior(labels, labels.shape[0])
        # TODO
        self.totalTrainingCount += features.shape[0]
        self.totalTrainingClassCount += classCount
        pass
