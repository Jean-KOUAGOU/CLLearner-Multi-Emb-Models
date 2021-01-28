# CLLearner-Multi-Emb-Models
The repository contains our implementation of a concept length learner in description logics

## To reproduce the results:
*-* Install ontolearn and all its dependencies;

*-* Clone this repository: git clone https://github.com/Jean-KOUAGOU/CLLearner-Multi-Emb-Models.git;

*-* Choose the knowledge you want in main.py by inputting the right path. Not that the family\_forte knowledge graph is a small/toy data set, and should not be used here. Please feel free to add another big knowledge graph to run new experiments. It may be necessary to reduce path\_length and num\_generation_paths in the DataTriples arguments, especially on big knowledge graphs.;

*-* One can also choose the embeddings he/she wants, by default, ConEx embeddings will be used;

*-* If GPU is available, set train_on_gpu to True;

*-* In the train-test split line, choose test_size = 0.05;

*-* Set alpha = 1 or alpha = 0;

*-* In train arguments, choose kf\_n\_splits = 10 for a ten-fold cross-validation setting

*-* Save and run main.py;

*-* When prompted for the number of iterations, enter 1000 (if GPU is available);

*-* When prompted for a value for print_every, input an integer, which represents the inverse-frequency at which training details are shown, and controls the smoothness of training curves. 
