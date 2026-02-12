import tempfile
import subprocess

from tqdm import tqdm
from collections import Counter
import os
import ijson
from typing import List, Tuple, Dict, Optional, Any, Callable, Union
import json
from collections.abc import Iterable
import gc
import etree
import re

download_wsd = '''\
FILE=WSD_Training_Corpora
if [ -d "$FILE" ]; 
then 
    echo "$FILE exists."
else
    echo "$FILE not found. Downloading .."
    wget -O "$FILE.zip" http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip 2>/dev/null
    echo "$FILE download completed."
    unzip "$FILE.zip"
    echo "$FILE unzip completed."
    rm "$FILE.zip"
fi
FILE=WSD_Unified_Evaluation_Datasets
if [ -d "$FILE" ]; 
then 
    echo "$FILE exists."
else
    echo "$FILE not found. Downloading .."
    wget -O "$FILE.zip" http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip 2>/dev/null
    echo "$FILE download completed."
    unzip "$FILE.zip"
    echo "$FILE unzip completed."
    rm "$FILE.zip"
fi
'''

def run_script(script):
    with tempfile.NamedTemporaryFile() as scriptfile:
        scriptfile.write(script)
        scriptfile.flush()
        subprocess.call(['/bin/bash', scriptfile.name])

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

def dumb(i):
    """
    This is a fake Callable
    """
    return i
    
def parallelize_loop(func_: Callable, iter_: Iterable) -> List:
    import multiprocessing as mp 
    import signal
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(mp.cpu_count(),init_worker)
    output = []
    try:
        for sample in tqdm(pool.imap(func_, iter_), desc="parallel processing: "):
            output.append(sample)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        print('keyboard interruption')
    print('main process exiting..')
    pool.terminate()
    pool.join()
    return output

def iterate_xml(xmlfile):
    """
    Iterate over xml tags and yield a tag each time the closing is encountered
    Yields: <sentence></sentence> DOM elements
    """
    doc = etree.iterparse(xmlfile, events=('start', 'end'))
    _, root = next(doc)
    start_tag = None
    for event, element in doc:
        if event == 'start' and start_tag is None:
            start_tag = element.tag
        if event == 'end' and element.tag == start_tag:
            yield element
            start_tag = None
            element.clear()
    root.clear()

def build_fake_xml(path: str, path_fake: str) -> None:
    """
    Create a one root SemCor+OMSTI dataset
    """
    try:
        # read semcor+omsti.data.xml
        with open(path) as f:
            print('Semcor and OMSTI file found. Reading..')
            xml = f.read()
        # since there are multiple root, delete them and create a unique fake root
        xml_fake = re.sub(r"(<\?xml[^>]+\?>)", r"\1\n<root>", xml) + "</root>"
        xml_fake = re.sub(r"(<corpus[^>]+>\n)", r"", xml_fake)
        xml_fake = re.sub(r"</corpus>\n", r"", xml_fake)
        xml_fake = re.sub(r"(<text[^>]+>\n)", r"", xml_fake)
        xml_fake = re.sub(r"</text>\n", r"", xml_fake)
        # write the customized dataset
        with open(path_fake,'w') as f:
            print('Semcor and OMSTI fake root file created. Writing..')
            f.write(xml_fake)
        del xml_fake
        del xml
        gc.collect()

    except OSError as err:
        print("OS error, file ::semcor+omsti.data.xml:: not present in specified path: {0}".format(err))

def semcor_omsti_parser(dir_path: str, name: str = 'semcor+omsti')->List[Dict]:
    """
    Parse the Semcor+OMSTI XML file dataset
    Return: List of Dict of samples with fields:

    """
    from nltk.corpus import wordnet

    new_name = name+'+synsets'
    path_dataset = os.path.join(dir_path, new_name+'.json')
    json_dataset = list(dict())
    if not os.path.isfile(path_dataset):
        print('JSON dataset not found. Parsing..')
        # prepare the xml file in order to iterate over it using lxml.etree
        path = os.path.join(dir_path,name+'.data.xml')
        path_fake = os.path.join(dir_path,name+'.data.fake.xml')
        if not os.path.isfile(path_fake):
            print('Creating unique root for '+name+' dataset. Parsing..')
            build_fake_xml(path,path_fake)
        else:
            print(name+' fake root file found.')
        
        # at this point read the keys and allocate correspondance in a dict 
        path_key = os.path.join(dir_path,name+'.gold.key.txt')
        id_to_wordnet = dict()
        try:
            with open(path_key) as f:
                print(name+' gold keys found. Reading..')
                for line in f:
                    key, *value = line.split()
                    id_to_wordnet[key] = value
        except OSError as err:
            print("OS error, file ::"+name+".gold.key.txt:: not present in specified path: {0}".format(err))
        
        # iterate over yielded sentences and collect samples
        gc.collect()
        print('JSON dataset building..')
        json_dataset = []
        i=0
        for sentence in tqdm(iterate_xml(path_fake), desc="parsing samples: "):
            instances_index_id = []
            tokens = []
            lemmas = []
            pos = []
            # collect each instance in sentence and create a new sample for each instace
            for index, word in enumerate(sentence):
                if word.tag == 'instance':
                    instances_index_id.append((index,word.attrib['id']))
                # parse the whole sentence
                tokens.append(word.text)
                lemmas.append(word.attrib['lemma'])
                pos.append(word.attrib['pos'])
            # generate the sample with all relevant data
            for index_instance, id_instance in instances_index_id:
                sample = dict()
                key_instance = id_to_wordnet[id_instance]
                lemma = wordnet.lemma_from_key(key_instance[0])
                synset = lemma.synset()
                all_senses = wordnet.lemmas(lemma.name(), lemma.synset().pos())
                sample['instance_id'] = id_instance # Semcor + OMSTI dataset instance id e.g. 'd000.s046.t005'
                sample['wordnet_key'] = key_instance # WordNet style keys e.g. 'public%5:00:00:common:02'
                sample['word_index'] = index_instance
                # sample['tokens'] = tokens
                sample['lemmas'] = lemmas
                sample['pos'] = pos
                sample['synsets'] = [sense.synset().name() for sense in all_senses]
                sample['glosses'] = [sense.synset().definition() for sense in all_senses]
                sample['examples'] = [sense.synset().examples() for sense in all_senses]
                sample['label'] = synset.name()
                json_dataset.append(sample.copy())

        with open(path_dataset, 'w') as f:
            json.dump(IteratorAsList(iter(json_dataset)), f)

        for sample in json_dataset[:2]:
            print("*** Example ***")
            print("sample: %s" % sample)

        print('Done.')
        gc.collect()

    else: 
        
        print('Dataset already present!')

if __name__ == "__main__":
    run_script(download_wsd)

    semcor_omsti_parser(dir_path = 'WSD_Training_Corpora/SemCor+OMSTI/',name = 'semcor+omsti')

    semcor_omsti_parser(dir_path = 'WSD_Unified_Evaluation_Datasets/ALL',name = 'ALL')