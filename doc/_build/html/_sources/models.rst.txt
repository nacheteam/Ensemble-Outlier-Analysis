Models
========


----



Base Model
------------------

All the models are coded starting from the same class. This class is called EnsembleTemplate.

The implementation is available in the link: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/models/base.py

The class contains 6 methods as the skeleton of the rest of the classes. Those methods
are:

1. "__init__": The method init is the constructor of the class and has the contamination parameter as default.
This parameter indicates which percentage of the dataset is considered as outliers. If 0.1 is passed then the 10% of the dataset is considered as outliers.

2. "fit": This method is the main mathod for the user. The method needs the dataset as a parameter. The dataset is stored and the outlier scores are initialized
an finally the method "runMethod" is executed so the model can update the scores.

3. "runMethod": This is the method where the actual model is coded. This method is only supposed to be called from the "fit" method and not from outside
of the class.

4. "getRawScores": This method just returns the scores.

5. "getOutliersBN": This method receives as a parameter the number of outliers and with that parameter the method returns the indexes of the instances
with the highest score value.

6. "getOutliers": This method takes the contamination parameter and returns the percentage with the highest scores corresponding to that percentage.


----

HICS
------------------

The implementation of this model is taken from the paper High Contrast Subspaces for Density-Based Outlier Ranking written by Fabian Keller, Emmanuel Müller and Klemens Böhm.

The code itself can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/models/hics.py

The parameters of the model are the following ones:

- outlier_rank: this is the method used to evaluate the instances in the las step of the algorithm. The original proposal used LOF but any density-based method
can be used as well. The options available are: LOF, COF, CBLOF, LOCI, HBOS and SOD. Default is LOF.

- M: This is the number of times subsampling is applied in the process of obtaining the contrast of a subspace. Default is 100.

- alpha: The parameter that computes the size of the test sample when computing the contrast. Default is 0.1.

- numCandidates: For each dimension only numCandidates subspaces are retained until the end of the algorithm. Default is 500.

- maxOutputSpaces: At the end of the algorithm this is the máximum number of subspaces returned as high contrast ones. Default is 1000.

- numThreads: Number of threads for the parallel code execution.

- verbose: boolean parameter to indicate if the algorithm should print the progress.

Now we are going to explain the basis of the model. The model goes through all dimensions from 2 to the maximum dimensionality. For each dimension the high contrast subspaces
are computed. All possible subspaces are tried on dimension 2 and for higher dimensions only the high contrast ones are used as fathers of the next subspaces.

For example if [0,2] is a high contrast subspace then the childs or candidates for dimension 3 could be for example [0,2,3], [0,2,1], [0,2,5], etc.

To compute the constrast we take a subsample of the dataset using alpha to compute the size of this sample. Over this sample the deviation is computed. This procedure works as follows:
for each instance in the dataset and for a fixed comparison atribute we make the cumulative sum of all the elements in the sample if the value of the comparison attribute
of the instance and each instance of the dataset is bigger. We make the same cumulative value but considering only the sample. Finally the absolute value of the difference
is computed. Now for each instance we have a value. We take the maximum of these differences as the deviation.

Finally after we have all the high contrast subspaces then LOF or outlier_rank method is applied in each projection using the subspaces. Then the
results are averaged to obtain the final scores.

----

Mahalanobis Kernel
------------------

The implementation is picked from Aggarwal, Charu C., Sathe, Saket Outlier Ensembles book. The model is parameter free so no adjusting is required.

The code itself can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/models/KernelMahalanobis.py

Let's describe the way this method works. First we take the data as a matrix and compute the similarity matrix :math:`S = DD^T`. Then Singular Value Decomposition
is applied to this matrix and we obtain :math:`S = Q\Delta Q^T` so we can compute :math:`D' = Q\Delta` and scale it to zero mean and unit variance.

Then with this matrix as a result we can obtain it's mean row vector and compute the anomaly scores as the Euclidean distance of each row to the mean.

----

LODA
------------------

The implementation is picked from LODA: Lightweight on-line detector of anomalies written by Tomas Pèvny.

The code can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/models/loda.py

The model has 2 parameters:

- k: this is the number of projections and therefore the number of histograms computed default is 500.
- n_bin: number of bins of the histogram default is 25.

The model creates first the random projection vectors. These vectors are as long as each instance of the dataset and contains :math:`[\sqrt{d}]` where :math:`d` is the
dimensionality of the dataset itself. The rest of elements of the projection vector are taken from a normal distribution with zero mean and unit variance.

Using this procedure k proection vectors are created. Now for each one of this vectors we are going to compute the one-dimensional projection for each instance
of the dataset like :math:`z_j = x_j \cdot w_i^T` where :math:`x_j` is an instance of the dataset and :math:`w_i` a projection vector. With these values a histogram
is computed so we can end up with k histograms.

Finally with these histograms we can compute for each instance the probability of appearance of the instance looking for its corresponding bin in each histogram. Then,
with these :math:`k` probability values we can compute the final score with the formula :math:`score = - \sum_{i=1}^{k}\frac{log(p_i)}{k}`.

----

OUTRES
------------------

The implementation of this model is taken from OUTRES: Statistical Selection of Relevant Subspace Projections for Outlier Ranking written by Emmanuel Müller, Matthias Schiffer and Thomas Seidl.

The code can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/models/outres.py

This model has a single parameter:

- alpha: this parameter is the level of confidence for the Kolmogorov-Smirnov test to check is a subspace is relevant or not. Default is 0.01.

- verbose: parameter that indicates if the progress should be printed.

- experiment: parameter that indicates if the second experiment is being run. This only makes the model save the internal information to files.

This model evaluates the score instance by instance. For each instance it starts checking for relevant subspaces of dimension 2. We say a subspace
is relevant if the 1D projections included in the subspace projection are not uniformly distributed. The rest of subspaces with higher dimensionality are
created or proposed starting from lower dimensionality subspaces that have been relevant.

If the subspace is relevant then we can compute the density of the instance. This is checking in the neighborhood if the projection of the data is nearby our instance.
The density is computed with the formula :math:`\frac{1}{n}\sum_{p\in AN(o,S) K_e (\frac{dist_S (o,p)}{\epsilon(|S|)})}` where :math:`o` is the instance being scored,
:math:`AN(o,S)` is the neighborhood of the instance :math:`o` in the subspace :math:`S`, :math:`dist_S (o,p)` is the distance from the instance :math:`o` to the
instance :mat:`p` in the projection of the data, :math:`\epsilon(|S|)` is a measure to make the adaptative neighborhood and :math:`K_e` is the function
:mat:`K_e (x) = 1-x^2`.

The adaptative neighborhood are the instances that are at the most at distance :math:`\epsilon (|S|)` to our instance. The calculation of :math:`\epsilon (|S|)` can be
found in the paper.

With the density computed we can compute the deviation as :math:`dev(o,S) = \frac{\mu - den(o,S)}{2\sigma}` where :math:`\mu` and :math:`\sigma` are the mean and
standard deviation of the density values in the adaptative neighborhood.

Finally if the deviation is bigger than one we update the score :math:`r(o) = r(o) \cdot \frac{den(o,S)}{dev(o,S)}`. The scores are in the interval :math:`[0,1]`
being 0 very outlying and 1 inlying. To mantaing the sacale (bigger is more outlying) we modify the scores at the end as :math:`1-r(o)`. Doing this the inliers
will have values nearby :math:`0` and outliers will have values nearby :math:`1`.

This process is repeated for every instance.

----

TRINITY
------------------

The implementatio  of this model is taken from the book Aggarwal, Charu C., Sathe, Saket Outlier Ensembles.

The code can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/models/trinity.py

This model has one single parameter:

- num_iter: This is the number of times the subsamping technique is used in each component of the model.

- verbose: boolean value indicating if the progress should be printed.

The model has three main modules: the distance-based, the dependency-based and the density-based.

In the distance-based module we use the KNN with :math:`k=5` algorithm for outlier detection having at the end a list with all the scores.

In the density-based module we use the IForest algorithm for outlier detection ending up with another list of scores.

In the dependency-based module we use the Mahalanobis Kernel method ending up with the third list of scores.

In all these three modules the subsampling technique is applied. This means that we don't use the whole dataset to fit the model.
we use only a small sample and repeat this process. At the end all instances will have their score. This technique allow us to reduce the variance.

The three lists of scores are scaled to zero mean and unit variance, then averaged to obtain the final score list.
