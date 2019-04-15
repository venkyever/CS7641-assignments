# CS7641: Machine Learning W2019 - Assignment 4
# Assignment 4 - Markov Decision Processes

Please pull the code from github: https://github.com/matanhalevy/CS7641-assignments and go to assignment 4 folder.

Due to issues importing a enterprise uploaded branch locally, I had to root Geoff Van Allmen's modificaitons to pymdptoolbox in my code, if there are any issues regarding this please clone it and add to assignment 4 folder.
https://github.gatech.edu/gva3/pymdptoolbox-cs7641, pleas

## General
You must have numpy, pymdptoolbox (branched version), scikit-learn, matplotlib, and basic python libraries installed to use this codebase.

To install any potential missing packages or incompaitable versions please in the root folder of the project run:

```
pip install -r 'assignment4\requirements.txt'
```


## Output

Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders are made for each learner but not environment, the files are labelled appropriately so one can navigate through.
If these folders do not exist the experiments module will attempt to create them.

## To Run Project:

```
python main.py -a True
```
Will run entire project,

Additionally you can choose to run certain sections with the -l tag and options of 'q_learner', 'policy_iter', 'value_iter', and 'plot'