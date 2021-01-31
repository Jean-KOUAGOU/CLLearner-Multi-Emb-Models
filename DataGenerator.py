from ontolearn import KnowledgeBase, LengthBasedRefinement
from conceptgenerator import CustomLearningProblemGenerator
from collections import defaultdict, Counter
import random, os, copy, numpy as np, pandas as pd
class DataTriples:
    """
    This class takes an owl file, loads it using ontolearn.base.KnowledgeBase resulting in a knowledge base.
    A refinement operator is used to generate new concepts of various lengths.
    The knowledge base is then converted into triples of the form: individual_i ---role_j---> concept_k and stored in a txt file (train.txt).
    The lengths and the respective positive and negative examples of each concept are also stored in dedicated dictionaries.
    
    """

    def __init__(self, path='', num_generation_paths=10, path_length=12, num_of_concept_per_length=80, min_child_length=2, num_ex=1000, concept_redundancy_rate=0.):
        self.path = path
        kb = KnowledgeBase(path=path)
        self.concept_redundancy_rate = concept_redundancy_rate
        self.kb = kb
        self.num_ex = num_ex
        self.atomic_concepts = list(kb.get_all_concepts())
        self.atomic_concept_names = set([a.str for a in list(kb.get_all_concepts())])
        rho = LengthBasedRefinement(kb)
        self.lp_gen = CustomLearningProblemGenerator(knowledge_base=kb, refinement_operator=rho, num_problems=num_generation_paths, depth=path_length, min_length=min_child_length)
    
    def set_num_of_concepts_per_length(self, l):
        self.num_of_concepts_per_length = l
        
    def __base_path(self, path):
        for i in range(len(path))[::-1]:
            if path[i] == "/":
                return i
    def kb_to_triples(self, export_folder_name='Triples'):
        self.concept_pos_neg = defaultdict(lambda: defaultdict(list))
        self.concept_lengths = defaultdict(float)
        
        if not os.path.exists(os.path.join(self.path[:self.__base_path(self.path)], export_folder_name)):
            os.mkdir(os.path.join(self.path[:self.__base_path(self.path)], export_folder_name))
            train_file = open("%s/train.txt" % os.path.join(self.path[:self.__base_path(self.path)], export_folder_name), mode="w")
            non_isolated_file = open("%s/non_isolated.txt" % os.path.join(self.path[:self.__base_path(self.path)], export_folder_name), mode="w")
            non_isolated_individuals = set()
            for rel in self.kb.property_hierarchy.all_properties:
                for tple in rel.get_relations():
                    train_file.write(str(tple[0])+"\t\t"+str(rel)+"\t\t"+str(tple[1])[:50]+"\n")
                    non_isolated_individuals.update([str(tple[0]), str(tple[1])])
            for indiv in non_isolated_individuals:
                non_isolated_file.write(str(indiv)+"\n")
            train_file.close(); non_isolated_file.close()
        else:
            non_isolated_individuals = open(os.path.join("./"+self.path[:self.__base_path(self.path)], export_folder_name)+"/"+"non_isolated.txt", "r")
            non_isolated_individuals = non_isolated_individuals.read()
            non_isolated_individuals = non_isolated_individuals.split("\n")
            print("Example of non isolated individual: ", non_isolated_individuals[0])
        
        self.concept_length_dist = Counter()
        All_individuals = set(self.kb.get_all_individuals())
        print("Number of individuals in the knowledge base: {} \n".format(len(All_individuals)))
        
        Nodes = set(self.lp_gen)
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concepts))
        print("Longest concept length: ", np.max([len(n) for n in Nodes]), "\n")
        print("Total number of new concepts generated: ", len(Nodes), "\n")
        self.train_concepts = []
        Concepts = {c.str: c for c in ([node.concept for node in Nodes] + self.atomic_concepts)}.values()
        total_concepts = len(Concepts)
        No_concept_redundancy_map = dict()
        No_redundancy_length_counts = Counter()
        for i, concept in enumerate(Concepts):
            valid_neg = sorted(set(pd.Series(list(All_individuals-concept.instances)).apply(lambda x: str(x))).intersection(set(non_isolated_individuals)))
            valid_pos = sorted(set(pd.Series(list(concept.instances)).apply(lambda x: str(x))).intersection(set(non_isolated_individuals)))
            if (i+1)%500 == 0:
              print("Progression: {}%".format(round(100.*(i+1)/total_concepts, ndigits=2)))
            if min(len(valid_neg),len(valid_pos)) >= self.num_ex//2:
              num_pos_ex = self.num_ex//2
              num_neg_ex = self.num_ex//2
            elif len(valid_pos) >= len(valid_neg) and len(valid_pos) + len(valid_neg) >= self.num_ex:
              num_neg_ex = len(valid_neg)
              num_pos_ex = self.num_ex-num_neg_ex
            elif len(valid_pos) + len(valid_neg)>=self.num_ex:
              num_pos_ex = len(valid_pos)
              num_neg_ex = self.num_ex-num_pos_ex
            else:
              continue

            positive = list(random.sample(valid_pos, num_pos_ex))#valid_pos[:num_pos_ex]
            negative = list(random.sample(valid_neg, num_neg_ex))#valid_neg[:num_neg_ex]
            if self.concept_length_dist[concept.length] < self.num_of_concepts_per_length:
                instance_statistics = {atomic: 0 for atomic in self.atomic_concept_names}
                for ind in concept.instances:
                  types = set([str(t).split(".")[-1] for t in ind.is_a])
                  for t in types.intersection(self.atomic_concept_names):
                    instance_statistics[t] += 1
                instance_statistics.update({"num_pos_examples": len(concept.instances)})
                self.concept_length_dist.update([concept.length])
                if not concept.str in self.concept_pos_neg:
                  self.concept_pos_neg[concept.str]["positive"] = positive
                  self.concept_pos_neg[concept.str]["negative"] = negative
                  self.concept_pos_neg[concept.str]["stats"] = list(instance_statistics.values())
                rand = random.random()
                if str(valid_pos) in No_concept_redundancy_map and No_concept_redundancy_map[str(valid_pos)].length > concept.length:
                  No_concept_redundancy_map[str(valid_pos)] = concept
                  No_redundancy_length_counts.update([concept.length])
                elif (str(valid_pos) in No_concept_redundancy_map and\
                No_redundancy_length_counts[concept.length] < max(No_redundancy_length_counts.values())*self.concept_redundancy_rate):#and No_concept_redundancy_map[str(valid_pos)].length > concept.length:
                  No_concept_redundancy_map[str(valid_pos)+str(random.random())] = concept
                  No_redundancy_length_counts.update([concept.length])
                elif not str(valid_pos) in No_concept_redundancy_map:
                  No_concept_redundancy_map[str(valid_pos)] = concept
                  No_redundancy_length_counts.update([concept.length])

            
        self.No_concept_redundancy_map = No_concept_redundancy_map
        print("Data preprocessing ended successfully")

    def save_train_data(self):
      data = {"concept name": [], "positive examples": [], "negative examples": [], "pos ex stats": [], "concept length": []}
      for concept in self.No_concept_redundancy_map.values():
        data["concept name"].append(concept.str)
        data["positive examples"].append(self.concept_pos_neg[concept.str]["positive"])
        data["negative examples"].append(self.concept_pos_neg[concept.str]["negative"])
        data["pos ex stats"].append(self.concept_pos_neg[concept.str]["stats"])
        data["concept length"].append(concept.length)
      pd.DataFrame(data).to_csv("./"+("/").join(self.path.split("/")[1:-1])+"/"+"data.csv")
      print("Data saved at %s"% "/"+("/").join(self.path.split("/")[1:-1]))
