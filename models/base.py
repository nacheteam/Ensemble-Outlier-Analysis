class EnsembleTemplate:
    '''
    Template class for the ensemble anomaly detectors.
    '''

    def __init__(self):
        '''
        Init template
        '''
        pass

    def fit(self, dataset):
        '''
        Function to set the dataset
        '''
        self.dataset = dataset
        return self

    def runMethod(self):
        '''
        Function to run the method implemented
        '''
        self.outlier_score = np.array(list(range(len(self.dataset))))
        self.outliers = list()

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
