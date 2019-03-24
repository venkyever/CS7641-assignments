# CS7641: Machine Learning W2019 - Assignment 3
# Assignment 3 - Unsupervised Learning and Dimensionality Reduction

Please pull the code from github: https://github.com/matanhalevy/CS7641-assignments and go to assignment 3 folder.

All code used in assignment 3 is credited to Chad Maron and copied directly with only minor modifications from: 
https://github.com/cmaron/CS-7641-assignments

## General
You must have numpy, scikit-learn, matplotlib, and basic python libraries installed to use this codebase.

To install any potential missing packages or incompaitable versions please in the root folder of the project run:

```
pip install -r 'assignment3\requirements.txt'
```

## Data

The data we use is under "./data/twitter_df.csv" and "./data/speed_dating_df.csv". It is the result of the data preparation script in assignment 1.

## Output

Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders are created for each sub-experiment (clustering, benchmark NN, pca, ica,
random projections (RP), random forest feature importance (RF)).

If these folders do not exist the experiments module will attempt to create them.

## Running Experiments (taken from chad's code)
1. Update `run_experiment.py` to use your data sets for dataset1 and dataset2. Also set `best_nn_params` for your data sets (lines 94 and 101).
2. Run the various experiments (perhaps via `python run_experiment.py --all`)
3. Plot the results so far via `python run_experiment.py --plot`
4. Update the dim values in `run_clustering.sh` based on the optimal values found in 2 (perhaps by looking at the scree graphs)
5. Run `run_clustering.sh`
6. One final run to plot the rest `python run_experiment.py --plot`

## Clustering Experiments

The experiments will output modified versions of the data sets after applying the DR methods. The script `run_clustering.sh` can be used to perform clustering on these modified datasets, using a specific number of components for the DR method.

**BE SURE TO UPDATE THE VALUES IN THIS SCRIPT FOR YOUR DATASETS**. 

There are different optimal values for each algorithm and each dataset, and using the wrong value will make you a sad panda.

## Graphing

The run_experiment script can be use to generate plots via:

```
python run_experiment.py --plot
```

Since the files output from the experiments follow a common naming scheme this will determine the problem, algorithm,
and parameters as needed and write the output to sub-folders in `./output/images`.