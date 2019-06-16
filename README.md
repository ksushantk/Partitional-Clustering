# Partitional-Clustering
Use of popular metaheuristic algorithms for finding optimal cluster centers during clustering


# Harmony Search based Clustering Algorithm
	- This set of files perform Harmony search based clustering algorithm.
	- The proposed novel partitional clustering approach extracts information in the form of optimal cluster centers from training samples. The extracted cluster centers are then validated on test samples.
	- This illustration contains two files, namely, Main_fn.m and Harmony_Search.m 
	- Main_fn.m is the main file which generates synthetic data. Post the training phase, clustering is carried out on test data set and results are displayed.
	- Harmony_Search.m contains optimal cluster center extraction from the training dataset using Harmony search. The file takes in training dataset, upper & lower limits of data and number of attributes as the input and returns the optimal cluster center to the Main_fn.m.
	- In the main file, a synthetic data is generated with predefined mean and standard deviation. The users can vary these parameters. The users can also implement algorithm using there own datasets. The dataset and corresponding training, testing portion should replace the variables xdata, ftrain, ftest with section of lines 20-45 in main_fn.m
	- The result of clustering can be visualized through confusion matrix in line 122-124 and Overall Accuracy in lines 130-133 of Main_fn.m
	- Note: Please do not switch between figure windows during program execution.
	
# Bat Algorithm based Clustering Algorithm
	- The two matlab files, namely, Main.m and Bat_Algorithm.m are used to perform data clustering using Bat Algorithm.
	- The proposed novel partitional clustering approach extracts information in the form of optimal cluster centers from training samples. The extracted cluster centers are then validated on test samples.
	- Main.m is the file which needs to be executed. This loads the dataset and extract the cluster centers using training dataset. Post the training phase, clustering is carried out on test dataset and results are displayed.
	- Bat_Algorithm.m is the file called from Main.m for extracting the optimal cluster centers from training dataset using Bat Algorithm. The file takes in training dataset with the upper & lower limits from each attributes as the input to the algorithm. The file returns the optimal cluster center to the Main.m
	- In this illustration, a synthetic data is generated with predefined mean and standard deviation. The users can vary these parameters. If the users want to test on there own datasets, then the dataset have to be segregated into the corresponding training and testing portion. The lines 20 to 53 needs to be modified accordingly by assigning related dataset to the variables xdata (complete dataset), ftrain (traning dataset) and ftest (testing dataset) in the file Main.m
	- The result of clustering can be visualized in the command prompt through the confusion matrix.
	
# Flower Pollination Algorithm based Clustering Algorithm
	- The two Matlab files, namely, Data_Clustering_FPA.m and Flower_Pollination.m are used to perform data clustering using Flower Pollination Algorithm.
	- The proposed partitional clustering approach extracts information in the form of optimal cluster centers from training samples. The extracted cluster centers are then validated on test samples.
	- Data_Clustering_FPA.m is the file which needs to be executed. This generates a synthetic dataset with predefined mean and standard deviation. Data is divided into train and test with predefined ratio. The users can vary these parameters. Training data is used to extract the cluster centers. Post the training phase, clustering is carried out on test dataset and results are displayed.
	- Flower_Pollination.m is the file called from Data_Clustering_FPA.m for extracting the optimal cluster centers from training dataset using Flower Pollination Algorithm. The file takes in training dataset with the upper & lower limits for each attribute as the input to the algorithm. The file returns the optimal cluster center to the Data_Clustering_FPA.m
	- In this illustration, a synthetic data with two class and two attributes is employed. The users can use their own custom datasets by replacing this synthetic data generation code using their own datasets. The users can vary FPA parameters based on the datasets they use for clustering.
	- The result metrics (optimal cluster centers, confusion matrix and classification error percentage) of clustering is displayed on the terminal. Further train, test data, initial agents of flower pollination and subsequent movement of agents to optimize are also visualized.
	