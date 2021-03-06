class EnsembleTemplate:
    '''
    Template class for the ensemble anomaly detectors.
    '''

    def __init__(self, contamination=0.1):
        '''
        Init template
        '''
        pass

    def fit(self, dataset):
        '''
        Function to set the dataset and execute the algorithm
        '''
        self.dataset = dataset
        self.outlier_score = [0]*len(self.dataset)
        self.outliers = []
        self.runMethod()
        return self

    def runMethod(self):
        '''
        Function to run the method implemented
        '''
        pass

    def getRawScores(self):
        '''
        Function that gets the raw scores
        '''
        return self.outlier_score

    def getOutliersBN(self, noutliers):
        '''
        Function that gets the noutliers instances of the most outlying data
        '''
        return self.outliers

    def getOutliers(self):
        pass
