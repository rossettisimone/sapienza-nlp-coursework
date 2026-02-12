import numpy as np
from typing import List, Tuple, Dict

from model import Model
from nltk.corpus import wordnet as wn
import nltk
nltk.download("wordnet")
import pytorch_lightning as pl
import torch
from .wsdwic import WiCWSDModel, HParams, collate_fn, to_sample

pl.seed_everything(41296)

mapping = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

SYNSETS = list(wn.all_synsets())

SYNSETS_TO_IDS = { synset.name(): index for index, synset in enumerate(SYNSETS) }

IDS_TO_SYNSETS = { index: synset_name for synset_name, index in SYNSETS_TO_IDS.items() }


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device) #RandomBaseline()


class RandomBaseline(Model):

    def __init__(self):
        # Load your models/tokenizer/etc that only needs to be loaded once when doing inference
        pass

    def predict(self, sentence_pairs: List[Dict]) -> Tuple[List[str], List[str]]:
        preds_wsd = [(np.random.choice(wn.synsets(pair["lemma"], mapping[pair["pos"]])).lemmas()[0].key(), \
            np.random.choice(wn.synsets(pair["lemma"], mapping[pair["pos"]])).lemmas()[0].key()) for pair in sentence_pairs]
        preds_wic = []
        for pred in preds_wsd:
            if pred[0] == pred[1]:
                preds_wic.append('True')
            else:
                preds_wic.append('False')
        return preds_wic, preds_wsd


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    
    def __init__(self, device):
        # Load your models/tokenizer/etc that only needs to be loaded once when doing inference
        self.wsdwid = WiCWSDModel(HParams(),device)
        checkpoint = torch.load('./model/wic-wsd-epoch=09-wic_val_Accuracy=0.67.ckpt',map_location=device)
        self.wsdwid.load_state_dict(checkpoint['state_dict'])
        # assert Bert if freeze
        for par in self.wsdwid.parameters():
            par.requires_grad = False
        self.wsdwid.eval()

    def predict(self, sentence_pairs: List[Dict]) -> Tuple[List[str], List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!

        sample_pairs = to_sample(sentence_pairs)
        batch = collate_fn(sample_pairs)

        with torch.no_grad():
            result = self.wsdwid.eval().predict(batch)
        preds_wsd = [tuple([wn.synset(IDS_TO_SYNSETS[i]).lemmas()[0].key()\
             for i in res]) for res in result['wsd_pred']]
        preds_wic = ['True' if round(res)==1 else 'False' for res in result['wic_pred']]

        return preds_wic, preds_wsd
