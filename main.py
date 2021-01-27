import random, os, copy, torch, torch.nn as nn, numpy as np
from Modified_ConEx.main import Experiment
from Modified_ConEx.save_embeddings import save_ConEx_emb
# os.chdir('/content/drive/My Drive/ResearchProjectAMMI/ConEx/')
# from main import Experiment
from Modified_ConEx.helper_classes import Data
import rdflib
#from helper_classes import Data
from DataGenerator import DataTriples
from LearningApproach import *
import pandas as pd
from sklearn.model_selection import train_test_split

train_on_gpu = False
from sklearn.metrics import f1_score
seed = 1
random.seed(seed)
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./KGs/carcinogenesis/carcinogenesis.owl", nargs="?",
                            help="Which dataset to use ? Format ./KGs/xxx/yyy.owl")
    args = parser.parse_args()
    path = args.dataset
    dataset = "./"+("/").join(path.split("/")[1:-1])+"/"+"Triples"
    data_dir = "%s/" % dataset
    if not path.split("/")[-1].split(".")[0] in ["carcinogenesis", "mutagenesis", "NTNcombined"]:
        data_to_triples = DataTriples(path=path, num_generation_paths=10, path_length=5, min_child_length=2, num_ex=200, concept_redundancy_rate=0.)
        data_to_triples.set_num_of_concepts_per_length(300)
        data_to_triples.kb_to_triples()
        data_to_triples.save_train_data()

        torch.backends.cudnn.deterministic = True
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        d = Data(data_dir=data_dir, reverse=False, no_valid_and_test=True)
        experiment = Experiment(model='Conex', num_iterations=1000, batch_size=2000,
                                embedding_dim=20,
                                learning_rate=0.005, decay_rate=1., conv_out=16,
                                projection_size=272,
                                input_dropout=0.0, hidden_dropout=0.1,
                                feature_map_dropout=0.1, label_smoothing=0, cuda=True)
        experiment.train_and_eval(d, d.info, show_every=100)
        save_ConEx_emb(d, experiment, data_dir)

    Embeddings = pd.read_csv("./"+("/").join(data_dir.split("/")[1:-2])+"/"+"ConEx_emb.csv")
    Embeddings.set_index("Unnamed: 0", inplace=True)
    data = pd.read_csv("./"+("/").join(data_dir.split("/")[1:-2])+"/"+"data.csv")

    if path.split("/")[-1].split(".")[0] == 'carcinogenesis':
        data = data[data['concept length']!=5]
    print("Data size: ", data.shape[0])
    d_train, d_test = train_test_split(data, stratify=data['concept length'], test_size=0.2)

    alpha = 1.
    DATA = upsampling(d_train, 'concept length')
    data_X, data_y = conceptsInstances_to_numeric(DATA, Embeddings, with_stats=True, alpha=alpha)
    data_X, data_y = np.vstack(data_X[0]), np.array(data_y[0], dtype=int)

    X_test, Y_test = conceptsInstances_to_numeric(d_test, Embeddings, with_stats=True, alpha=alpha)
    X_test, Y_test = np.vstack(X_test[0]), np.array(Y_test[0], dtype=int)


    num_classes = int(max(data['concept length'])+1)

    CLL = ConceptLengthLearner_GRU(input_size=40, output_size=num_classes, hidden_dim=100, n_layers=2)
    n_epochs = int(input("Number of epochs: "))
    print_every = int(input("Print every: "))
    CLL = train(CLL, data_X, data_y, data_dir, train_on_gpu=train_on_gpu, batch_size=400, epochs=n_epochs, lr=0.001, print_every=print_every, kf_n_splits=3)

    for x, y in load_data(X=X_test, y=Y_test, batch_size=X_test.shape[0], shuffle=False):
        break
    if train_on_gpu:
        x, y = x.cuda(), y.cuda()
    output = CLL(x)
    print("Test F1 score: ", f1_score(output.argmax(1).cpu().tolist(), list(y), average='macro'))

