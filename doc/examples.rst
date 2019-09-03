Examples
============




For each model we have developed an example so we can see how it works.

----


HICS Example
--------------------


Full example: `hics_example.py <https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/examples/hics_example.py>`_

1. Import model

    .. code-block:: python

        from hics import HICS


2. Read the data to fit the model. The data should be in a numpy array:

    .. code-block:: python

      dataset,labels = utils.readDataAbalone()
      #dataset,labels = utils.readDataYeast()
      #dataset, labels = utils.readDataCancer()

3. Initialize a :class:`HICS` detector and fit the model

    .. code-block:: python

      hics = HICS(verbose=True, outlier_rank="lof", contamination=0.1, M=100, alpha=0.1, numCandidates=500, maxOutputSpaces=1000, numThreads=8)
      hics.fit(dataset)

4. Get the plot of the scores colouring the anomalies in red.

    .. code-block:: python

      import utils
      utils.obtainResults(hics)

    .. figure:: figs/hics_plot.png
      :alt: HICS Scatter Plot

5. Get some statistical information from the scores.

    .. code-block:: python

      utils.obtainResults(hics)

      # Get the outliers
      outliers = hics.getOutliers()

      # Print the labels of the outliers
      print(labels[outliers])

    .. figure:: figs/hics_statistics.png
      :alt: HICS Statistics

6. Check if the anomalies obtained by the model are also obtained by pyod.

    .. code-block:: python

      # Check if pyod models get the same anomalies
      print("Getting anomalies based on the voting system to check")
      cm, df = utils.checkAnomalies(dataset, outliers)
      print("Common ones: " + str(cm))
      print("Different ones: " + str(df))


    .. figure:: figs/hics_common_different.png
        :alt: Number of common and different outliers with HICS and PyOD

----


Mahalanobis Kernel Example
---------------------------------


Full example: `KernelMahalanobis_example.py <https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/examples/KernelMahalanobis_example.py>`_

1. Import model

    .. code-block:: python

        from KernelMahalanobis import KernelMahalanobis


2. Read the data to fit the model. The data should be in a numpy array:

    .. code-block:: python

      dataset,labels = utils.readDataAbalone()
      #dataset,labels = utils.readDataYeast()
      #dataset, labels = utils.readDataCancer()

3. Initialize a :class:`KernelMahalanobis` detector and fit the model

    .. code-block:: python

      kernel_mahalanobis = KernelMahalanobis()
      kernel_mahalanobis.fit(dataset)

4. Get the plot of the scores colouring the anomalies in red.

    .. code-block:: python

      import utils
      utils.obtainResults(kernel_mahalanobis)

    .. figure:: figs/MK_plot.png
      :alt: Mahalanobis Kernel Scatter Plot

5. Get some statistical information from the scores.

    .. code-block:: python

      utils.obtainResults(kernel_mahalanobis)

      # Get the outliers
      outliers = kernel_mahalanobis.getOutliers()

      # Print the labels of the outliers
      print(labels[outliers])

    .. figure:: figs/MK_statistics.png
      :alt: Mahalanobis Kernel Statistics

6. Check if the anomalies obtained by the model are also obtained by pyod.

    .. code-block:: python

      # Check if pyod models get the same anomalies
      print("Getting anomalies based on the voting system to check")
      cm, df = utils.checkAnomalies(dataset, outliers)
      print("Common ones: " + str(cm))
      print("Different ones: " + str(df))


    .. figure:: figs/MK_common_different.png
        :alt: Number of common and different outliers with Mahalanobis Kernel and PyOD

----


LODA Example
----------------------------


Full example: `loda_example.py <https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/examples/loda_example.py>`_

1. Import model

    .. code-block:: python

        from loda import LODA


2. Read the data to fit the model. The data should be in a numpy array:

    .. code-block:: python

      #dataset,labels = utils.readDataAbalone()
      dataset,labels = utils.readDataYeast()
      #dataset, labels = utils.readDataCancer()

3. Initialize a :class:`LODA` detector and fit the model

    .. code-block:: python

      loda = LODA(n_bins=25, k=500)
      loda.fit(dataset)

4. Get the plot of the scores colouring the anomalies in red.

    .. code-block:: python

      import utils
      utils.obtainResults(loda)

    .. figure:: figs/loda_plot.png
      :alt: LODA Scatter Plot

5. Get some statistical information from the scores.

    .. code-block:: python

      utils.obtainResults(loda)

      # Get the outliers
      outliers = loda.getOutliers()

      # Print the labels of the outliers
      print(labels[outliers])

    .. figure:: figs/loda_statistics.png
      :alt: LODA Statistics

6. Check if the anomalies obtained by the model are also obtained by pyod.

    .. code-block:: python

      # Check if pyod models get the same anomalies
      print("Getting anomalies based on the voting system to check")
      cm, df = utils.checkAnomalies(dataset, outliers)
      print("Common ones: " + str(cm))
      print("Different ones: " + str(df))


    .. figure:: figs/loda_common_different.png
        :alt: Number of common and different outliers with LODA and PyOD

----


OUTRES Example
----------------------------


Full example: `outres_example.py <https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/examples/outres_example.py>`_

1. Import model

    .. code-block:: python

        from outres import OUTRES


2. Read the data to fit the model. The data should be in a numpy array:

    .. code-block:: python

      #dataset,labels = utils.readDataAbalone()
      dataset,labels = utils.readDataYeast()
      #dataset, labels = utils.readDataCancer()

3. Initialize a :class:`OUTRES` detector and fit the model

    .. code-block:: python

      outres = OUTRES(verbose=True, alpha=0.01)
      outres.fit(dataset)

4. Get the plot of the scores colouring the anomalies in red.

    .. code-block:: python

      import utils
      utils.obtainResults(outres)

    .. figure:: figs/outres_plot.png
      :alt: OUTRES Scatter Plot

5. Get some statistical information from the scores.

    .. code-block:: python

      utils.obtainResults(outres)

      # Get the outliers
      outliers = outres.getOutliers()

      # Print the labels of the outliers
      print(labels[outliers])

    .. figure:: figs/outres_statistics.png
      :alt: OUTRES Statistics

6. Check if the anomalies obtained by the model are also obtained by pyod.

    .. code-block:: python

      # Check if pyod models get the same anomalies
      print("Getting anomalies based on the voting system to check")
      cm, df = utils.checkAnomalies(dataset, outliers)
      print("Common ones: " + str(cm))
      print("Different ones: " + str(df))


    .. figure:: figs/outres_common_different.png
        :alt: Number of common and different outliers with OUTRES and PyOD

----

TRINITY Example
-----------------------


Full example: `trinity_example.py <https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/examples/trinity_example.py>`_

1. Import model

    .. code-block:: python

        from trinity import TRINITY


2. Read the data to fit the model. The data should be in a numpy array:

    .. code-block:: python

      #dataset,labels = utils.readDataAbalone()
      dataset,labels = utils.readDataYeast()
      #dataset, labels = utils.readDataCancer()

3. Initialize a :class:`TRINITY` detector and fit the model

    .. code-block:: python

      trinity = TRINITY(verbose=True, alpha=0.01)
      trinity.fit(dataset)

4. Get the plot of the scores colouring the anomalies in red.

    .. code-block:: python

      import utils
      utils.obtainResults(trinity)

    .. figure:: figs/trinity_plot.png
      :alt: TRINITY Scatter Plot

5. Get some statistical information from the scores.

    .. code-block:: python

      utils.obtainResults(trinity)

      # Get the outliers
      outliers = trinity.getOutliers()

      # Print the labels of the outliers
      print(labels[outliers])

    .. figure:: figs/trinity_statistics.png
      :alt: TRINITY Statistics

6. Check if the anomalies obtained by the model are also obtained by pyod.

    .. code-block:: python

      # Check if pyod models get the same anomalies
      print("Getting anomalies based on the voting system to check")
      cm, df = utils.checkAnomalies(dataset, outliers)
      print("Common ones: " + str(cm))
      print("Different ones: " + str(df))


    .. figure:: figs/trinity_common_different.png
        :alt: Number of common and different outliers with TRINITY and PyOD
