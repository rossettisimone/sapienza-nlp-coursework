
from tqdm import tqdm
from collections import Counter
import os
import ijson
from typing import List, Tuple, Dict, Optional, Any, Callable, Union

import json

class IteratorAsList(list):
    """
    this iterator permits to fake json and write line by line the json file
    """
    def __init__(self, it):
        self.it = it
    def __iter__(self):
        return self.it
    def __len__(self):
        return 1

def load(path: str) -> Dict[str,Dict[str,str]]:
    list_important_keys = ['instance_id', 'word_index', 'lemmas', 'pos', 'synsets', 'label']
    samples = dict()
    generator = (row for row in ijson.items(open(path),'item'))
    for sample_dict in tqdm(generator, desc="parsing samples: "):
        samples[sample_dict['instance_id']]={k:v for k,v in sample_dict.items() if k in list_important_keys}
    print('\nDone.')
    return samples

def build_wic():

    if not os.path.isfile('WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti+wic.json'):

        path_key = os.path.join('WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt')
        id_to_key = dict()
        key_to_id = dict()
        id_to_index = dict()
        with open(path_key) as f:
            for index,line in tqdm(enumerate(f),desc='parsing gold keys: '):
                key, *value = line.split()
                id_to_key[key] = value
                id_to_index[key] = index
                for val in value:
                    try:
                        key_to_id[val].append(key)
                    except KeyError:
                        key_to_id[val] = [key]

        index_to_id = {v:k for k,v in id_to_index.items()}

        key_list = []
        for v in id_to_key.values():
            key_list.extend(v)

        key_to_freq = Counter(key_list)

        from nltk.corpus import wordnet as wn

        key_to_opposite = dict()

        key_to_name = dict()

        for key in tqdm(key_to_freq.keys(), desc='parsing frequencies: '):
            lemma = wn.lemma_from_key(key)
            key_to_name[key] = lemma.name()
            synset = lemma.synset()
            synset_list = wn.synsets(lemma.name(),pos=synset.pos())
            opposite_keys = []
            for synset_ in synset_list:
                if synset!=synset_:
                    synset_lemma = synset_.lemmas()
                    for lemma_ in synset_lemma:
                        opposite_keys.append(lemma_.key())
            assert key not in opposite_keys
            key_to_opposite[key] = opposite_keys

        print(list(key_to_name.items())[:2])

        max_neg_samples_per_couple = 9#11
        negative_dataset = set()
        list_keys = list(key_to_freq.keys())
        for key1 in tqdm(list_keys,desc='parsing negatives: '):
            for id1 in key_to_id[key1][:max_neg_samples_per_couple]:
                for key2 in key_to_opposite[key1]:
                    try:
                        for id2 in key_to_id[key2][:max_neg_samples_per_couple]:
                            if key_to_name[key1] == key_to_name[key2] and \
                                id1!=id2 and id1!=None and \
                                id2!=None and id1!=[] and id2!=[]:
                                negative_dataset.add(frozenset((id1,id2)))
                    except KeyError:
                        pass
        
        print('Total negative samples: ', len(list(negative_dataset)))

        max_pos_samples_per_couple = 20#25
        positive_dataset = set()
        list_keys = list(key_to_freq.keys())
        for key in tqdm(list_keys,desc='parsing positives: '):
            for id1 in key_to_id[key][:max_pos_samples_per_couple]:
                for id2 in key_to_id[key][:max_pos_samples_per_couple]:
                    try:
                        if id1!=id2 and id1!=None and \
                            id2!=None and id1!=[] and id2!=[]:
                        
                            positive_dataset.add(frozenset((id1,id2)))
                    except KeyError:
                        pass
        
        print('Total positive samples: ', len(list(positive_dataset)))

        union_dataset = positive_dataset.union(negative_dataset)
        intersect_dataset =  positive_dataset.intersection(negative_dataset)
        difference_dataset = union_dataset.difference(intersect_dataset)
        print('Total dataset samples: ', len(list(difference_dataset)))

        dataset = load('WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti+synsets.json')
        print('** Dataset Samples **')
        print(list(dataset.items())[:2])

        wic_dataset = []
        for ind, sample in enumerate(tqdm(list(difference_dataset), desc='parsing wic dataset: ')):
            try:
                couple = list(sample)
                sample1 = dataset[couple[0]]
                sample2 = dataset[couple[1]]
                pos1 = sample1['pos'][sample1['word_index']]
                pos2 = sample2['pos'][sample2['word_index']]
                assert pos1==pos2
                new_couple = {
                    "id" : ind,
                    "pos": pos1,
                    "sentence1": sample1['lemmas'],
                    "sentence2": sample2['lemmas'],
                    "index1": sample1['word_index'],
                    "index2": sample2['word_index'],
                    "synset1": sample1['label'],
                    "synset2": sample2['label'],
                    "synsets1": sample1['synsets'],
                    "synsets2": sample2['synsets'],
                    "label": 'True' if sample1['label']==sample2['label'] else 'False',
                    "id1": couple[0],
                    "id2": couple[1],
                }
                wic_dataset.append(new_couple)
            except KeyError as e:
                print(e)
                print(couple)
            except IndexError as e:
                print(e)
                print(couple)
            

        path_dataset = ('WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti+wic.json')
        with open(path_dataset, 'w') as f:
            json.dump(IteratorAsList(iter(wic_dataset)), f)
        
        print('Done!')
    
    else: 

        print('Dataset already present!')

if __name__ == "__main__":
    build_wic()