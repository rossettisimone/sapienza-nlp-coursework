#!/usr/bin/env python
# coding: utf-8

# # Homework 2 - Aspect-Based Sentiment Analysis (ABSA)
# 
# Student:  `Simone Rossetti` # `1900592`
# 
# NLP course A.A 2020-2021 `@DIAG (Sapienza, University of Rome)`
# 
# 
# 

# ## Intro

# ABSA aims to identify the aspect terms of given target entities and the sentiment expressed towards the respective terms.
# 
# Our task is about performing ABSA on a dataset made of restaurants and laptops reviews from [SemEval2014 Task 4](https://www.aclweb.org/anthology/S14-2004.pdf).
# 
# In this dataset ABSA is splitted in the following subtasks:
# 
#     a) aspect term identification 
#     b) aspect term polarity classification 
#     c) aspect category identification 
#     d) aspect category polarity classification 
# 

# - Example Subtask b
#     - Input: I love their pasta but I hate their Ananas Pizza. + {(pasta); (Ananas Pizza)}
#     - Output: {(pasta, positive); (Ananas Pizza, negative)}
# 
# - Example Subtask a+b
#     - Input: I love their pasta but I hate their Ananas Pizza.
#     - Output: {(pasta, positive); (Ananas Pizza, negative)}
# 
# - Example Subtask c+d
#     - Input: I love their pasta but I hate their Ananas Pizza.
#     - Output: {(food, conflict)}
# 
# - Example Task a+b+c+d
#     - Input: I love their pasta but I hate their Ananas Pizza.
#     - Output: {(pasta, positive); (Ananas Pizza, negative)} + {(food, conflict)}

# ## Setup

# First of all let's include all the libraries we need: 
# 
# NOTE: since we are working locally in a `Docker` environment we will assume all the needed dependancies have been already installed by running the `pip install -r requirement.txt` file.
# 
# We will use:
# 
# * [pytorch](https://pytorch.org/) of course,
# * [torchtext](https://pytorch.org/text/) a very useful library for text parsing,
# * [pytorch_lightning](https://www.pytorchlightning.ai/) to standardize and simplify the training procedure,
# * [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable) a useful lybrary which implements a statistical method named Conditional Random Fields which aims at maximizing the log likelihood of classifications over patterns.

# In[1]:


import os
import gc
import json
import torch
import random
import torchtext
import numpy as np
from torch import nn
from tqdm import tqdm
from torchcrf import CRF
from torchtext import data
import torch.optim as optim
import pytorch_lightning as pl
from collections import Counter
from dataclasses import dataclass
from torchtext.vocab import Vocab
from torchtext.vocab import Vectors
from abc import ABC, abstractmethod
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score as sk_f1_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import MetricCollection, F1, Recall, Precision
from transformers import AutoTokenizer, AutoModel, AutoConfig

# This is a method to ensure `reproducibility`: 
# 
# The computations will be deterministic based on the `seed`, that way results are the same also in different runs.

# In[3]:


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # will use only deterministic algorithms

# In the `data/` folder there are two datasets:
# *   `restaurants`,
# *   `laptops`
# with the same annotation scheme. 
# 
# Only `restaurants` is annotated for aspect category tasks.
# 

# The dataset is a `JSON` file where each entry is a dictionary with the keys "targets" and "text" (aspect terms tasks) and "category" (aspect category tasks):
# <br>
# <br>
# {<br>
#   &emsp;  "text": "I love their pasta but I hate their Ananas Pizza.", <br>
#    &emsp; "targets": \[<br>
#    &emsp;&emsp;&emsp;         \[13, 17\], "pasta", "positive"], <br>
#     &emsp;&emsp;&emsp;        \[36, 47\], "Ananas Pizza", "negative"]<br>
#     &emsp;    \], <br>
#   &emsp;  "categories": \[<br>
#   &emsp;&emsp;&emsp;          \("food", "conflict"\)<br>
#   &emsp;      \], <br>
# }

# The most common notation for sequence tagging is [IOB tagging](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)), which marks each word with a:
# * `B` = Beginning: the first word of the span of a named entity;
# * `I` = Inside: continuation of the span of a named entity (obviously, must be after a `B` label of the same type);
# * `O` = Outside: for any other word.
# 
# Since in general words are splitted within the text.
# 
# The possible categories are:
# * `anecdotes/miscellaneous`
# * `price`
# * `food`
# * `ambience`
# * `service`
# 
# The polarities for the terms and the categories are:
# * `positive`
# * `negative`
# * `neutral`
# * `conflict`

# At this point, let's define a `torchtext.vocab.Vocab` struct which holds our classes for simplicity, infact `Vocab` class has its own method which does direct and inverse mapping, `string <-> index`, by using methods `itos()` (index to string) and `stoi()` (string to index):

# In[7]:


# first we define some common tokens
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SEP_TOKEN = '<sep>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'

IOB = Vocab({
    PAD_TOKEN: 0,
    'O':1,
    'B':2, 
    'I':3,
    },specials=[PAD_TOKEN])

POLARITY = Vocab({
    PAD_TOKEN: 0,
    'positive': 1, 
    'negative': 2, 
    'neutral': 3, 
    'conflict': 4,
    },specials=[PAD_TOKEN])

CATEGORY = Vocab({
    PAD_TOKEN: 0,
    'anecdotes/miscellaneous': 1,
    'price': 2,
    'food': 3,
    'ambience': 4,
    'service': 5
    },specials=[PAD_TOKEN])

# At this point we can parse our dataset and create the `Sample` instances which will be easier to deal with when duilding our dataset, we will define an abstract class since we will perform different experiments which requires different ways of building the object:

# The `cache` decorator will help us in storing the output of a method to save computation time!

# In[8]:


def cache(method):
    """
    This decorator caches the return value of a method 
    so that results are not recomputed
    """
    method_name = method.__name__
    def wrapper(self, *args, **kwargs):
        self._cache = getattr(self, '_cache', {})
        if method_name not in self._cache:
            self._cache[method_name] = method(self, *args, **kwargs)
        return self._cache[method_name]
    return wrapper


# Let's define now the abstract class which represents a generic `ABSASample`:

# In[9]:


@dataclass
class ABSASample(ABC):
    text: str
    targets: Optional[List[Union[Tuple[int,int],str]]] = None
    categories: Optional[List[Tuple[str,str]]] = None

    @abstractmethod
    def process(self) -> Dict[str,torch.Tensor]:
        pass

    @abstractmethod
    def process_a(self) -> Dict[str,torch.Tensor]:
        pass
    
    @abstractmethod
    def process_b(self) -> Dict[str,torch.Tensor]:
        pass

    @abstractmethod
    def process_c(self) -> Dict[str,torch.Tensor]:
        pass
    
    @abstractmethod
    def process_d(self) -> Dict[str,torch.Tensor]:
        pass

# We extend the `PyTorch Dataset` class and define the necessary batching iteration in the collate function, our `Dataset` will store its samples:

# In[14]:


class ABSADataset(Dataset):

    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


# ## Wrappers (PyTorch Lightning)

# At this point we can define the remaining wrappers which will permit us to deal with `Pytorch Linghtening Framework`.  There are several advantages in using this it since the `pytorch_lightning` framework permits us to focus on our models and data preprocessing and get rid of standard code.

# ### LightningDataModule
# 
# `LightningDataModule` handles all the data that we will be using. 
# 
# Here we just need to set up our datasets and dataloaders, which will handle the data batching. We just need to feed them to the `Dataloaders` and set a `batch_size`.

# In[15]:


class ABSADataModule(pl.LightningDataModule):

    def __init__(self, collate_fn, training_file, valid_file, test_file, batch_size:int = 32, task: str = 'a'):
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        # load_list will manage the data parsing into `Sample` class
        self.train_samples = load_list(training_file,task) 
        self.valid_samples = load_list(valid_file,task) 
        self.test_samples = load_list(test_file,task) 
        

    def setup(self, stage=None):
        self.train_dataset = ABSADataset(self.train_samples)
        self.valid_dataset = ABSADataset(self.valid_samples)
        self.test_dataset = ABSADataset(self.test_samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=os.cpu_count()-1 or 1, 
        ) # we assign the available workers minus one

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=os.cpu_count()-1 or 1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=os.cpu_count()-1 or 1,
        )


# ### Configuration and Hyper Parameters wrapper
# 
# We define a simple class `Config` which wraps our hyperparameters passed as dictionary, which let us access it's values as if they were attributes of the class. In this way we have a more clean way to define the hyperparameters thanks to the `dict` class and a faster way to access it's values thanks the to `Config` wrapping, furthermore we implement a conversion method which permits to return the dict of hyperparameters which can be saved within the model using the `pl` method `save_hyperparameters()` within the `LighteningModule`. The method `save_hyperparameters()` do not support tensors thus `hparams()` of `Config()` will return only the hyperparameters. Since we are going to pass also the selected `nn.Module` we want to train by the key `model` in the `Config()` struct, it will also be discarded when `hparams()` is called.

# In[16]:


class Config:
    """Simple dictionary wrapper"""
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def hparams(self):
        return {k:v for k,v in self.__dict__.items()\
            if not isinstance(v,torch.Tensor)\
                and k != 'model'}


# ### LighteningModule
# 
# `LighteningModule` will handle our model, it is the main class which permits to define help functions required by `pl` to easily interface with the `Trainer` module. We will define a `LighteningModule` wrapper and pass the `Model` we want to train as parameter. In general each model has it's own forward and loss, if each model returns a standard output, e.g. made of loss, logits, preds, .. we can exploit this standardization to use one unique `LighteningModule` wrapper for any model.
# 

# In[18]:


class ABSAModule(pl.LightningModule):
    def __init__(self, configs, *args, **kwargs):
        super(ABSAModule, self).__init__(*args, **kwargs)
        """
        LighteningModule which wraps the torch.nn.Model passed in configs
        and deal with the Trainer class, we just need to define
        custom training_step and other abstract methods.
        Args:
            configs (Config): the model hyperparameters and configuration.
        """
        # don't save embeddings in the hyperparameters
        self.save_hyperparameters(configs.hparams())
        # the model to train, which should return 
        # 'loss','preds','labels'
        self.model = configs.model(configs)
        # Measures
        classes = self.hparams.num_classes
        ignore_index = self.hparams.ignore_index
        # instantiate a `MetricCollection` wrapper for 
        # validation and test time with F1, Recall, Precision
        metrics = MetricCollection([
            F1(num_classes=classes,average='macro',\
                ignore_index=ignore_index),
            Recall(num_classes=classes,average='macro',\
                ignore_index=ignore_index),
            Precision(num_classes=classes,average='macro',\
                ignore_index=ignore_index)])
        # will store the validation metrics
        self.val_metrics = metrics.clone(prefix='val_')
        # will store the test metrics
        self.test_metrics =  metrics.clone(prefix='test_')
        
    def forward(
        self, 
        batch: Dict[str,torch.Tensor]
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        This performs a forward pass of the model, as well as 
        returning the predictions and loss.
        """
        return self.model.forward(batch)

    @torch.no_grad()
    def predict(
        self, 
        batch: Dict[str,torch.Tensor]
        ) -> Dict[str,torch.Tensor]:
        """
        This performs a forward pass of the model during 
        inference time, returning the predictions,
        the @torch.no_grad() decorator disable the gradient
        tracking. This is an external method, not used by PL.

        COMMENT: since we are going to define the 'prediction 
        behaviour' in the self.model itself according to the presence
        or not of the 'labels' in the batch dict, we want to ensure
        that the 'labels' are not present in the dict.
        Accordingly the self.model will not compute any loss. 
        """
        batch.pop('labels', None) # remove labels if any
        return self.model.eval().forward(batch)

    def basic_step(
        self, 
        batch: Dict[str,torch.Tensor],
        mode: str = 'train'
        ) -> Dict[str,torch.Tensor]:
        """
        This method perform the basic step common to all
        phases and logs the loss.
        mode = <train,val,test>
        """
        summary = self.forward(batch)
        self.log(mode+'_loss', summary['loss'], prog_bar=True)
        return summary

    def training_step(
        self, 
        batch: Dict[str,torch.Tensor],
        batch_idx: int,
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        This runs the model in training mode mode, ie. activates 
        dropout and gradient computation. It defines a single
        training step.
        Very important for PL to return the loss that will be used
        to update the weights
        """
        return self.basic_step(batch)
    
    def validation_step(
        self, 
        batch: Dict[str,torch.Tensor],
        batch_idx: int,
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        This runs the model in eval mode, ie. sets dropout to 0 
        and deactivates grad. Needed when we are in inference mode.
        No @torch.no_grad() is needed since it is automatic in PL.
        """
        return self.basic_step(batch,'val')

    def test_step(
        self, 
        batch: Dict[str,torch.Tensor],
        batch_idx: int,
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        This runs the model in eval mode, ie. sets dropout to 0 and 
        deactivates grad. Needed when we are in inference mode.
        No @torch.no_grad() is needed since it is automatic in PL.
        """
        return self.basic_step(batch,'test')

    def configure_optimizers(self):
        """
        [Required by lightning]
        PL runs this at the beginning to initialize the optimizer
        """
        return optim.Adam(self.parameters(),lr=self.hparams.lr),
            # torch.optim.lr_scheduler

    def write_metrics_end(
        self,
        batch_parts: Dict[str,torch.Tensor],
        metrics: MetricCollection
        ):
        """
        Write metrics at end, support also multi GPUs
        """
        preds = batch_parts['preds']
        labels = batch_parts['labels']
        # preds = preds.view(-1)
        # labels = labels.view(-1)
        # compute the metrics
        results = metrics(preds,labels)
        # log the metrics
        self.log_dict(results, on_step=False, on_epoch=True, prog_bar=True)

    def training_step_end(
        self,
        batch_parts: Dict[str,torch.Tensor],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward(),
        supports also multi GPUs
        """
        return batch_parts['loss']

    def validation_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Fill the metrics
        """
        self.write_metrics_end(batch_parts,self.val_metrics)

    def test_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Fill the metrics
        """
        self.write_metrics_end(batch_parts,self.test_metrics)

# ## Task A

# **Task A** consists in tagging a sequence of tokens.
# 
# In general sequence tagging is performed following the [IOB tagging](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) paradigm, which marks each aspect term with a:
# * `B` = Beginning: the first word of the span of a named entity;
# * `I` = Inside: continuation of the span of a named entity (obviously, must be after a `B` label of the same type);
# * `O` = Outside: for any other word.

# ### 2 Methods
# 
# To accomplish this task we will use two different approaches:
# * The first one based on `Pre-Trained`  [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) `Word Embeddings` and `LSTM Sequence Encoding`
# * The second one based on [Transformers](https://huggingface.co/transformers/), thus `Contextualized Word Embeddings` and `LSTM Sequence Encoding`
# 
# Models will share similiar structure except from the way in which the words representation is created.

# At this stage we are ready for the model buildings!

# We will be using:

# **Embedding Layer**
# 
# It is a matrix with `vocab_size` rows and `embedding_dim` columns. Each row represents a word in the vocabulary.
# 
# Init: `embedding = nn.Embedding(vocab_size, embedding_dim)` 
# 
# Forward: `x_embeddings = embedding(x)`
# 
# where `x` is `torch.LongTensor()` type and `x_embeddings` is `torch.FloatTensor()` type.

# **LSTM Layer**
# 
# It's a layer implementing an LSTM neural network. An LSTM processes the input text (where words have been encoded with their embeddings) from **left to right** and, for each of them, provides a new representation that takes into account the left context.
# 
# Init: `lstm = nn.LSTM(input_size, hidden_size, num_layers, **kwconfigs)` 
# 
# Forward: `x_lstm = lstm(x)`
# 
# where `x` and `x_lstm` are `torch.FloatTensor()` type.
# 
# `x` if a tensor of shape `(sequence_len, batch_size, hidden_dim)` or `(batch_size, sequence_len, hidden_dim)` (must be specified using flag `batch_first = True`).

# We will be using:

# **Linear Classifier**
# 
# Finally, to assign each token an IOB tag, we classify the token representations, i.e., the outputs of the LSTM, with one of the possible output classes.
# 
# Init: `linear = nn.Linear(in_features, out_features)` 
# 
# Forward: `x_linear = linear(x)`
# 
# where `x` and `x_embeddings` are `torch.FloatTensor()` type.
# 
# 

# **Conditional Random Fields Layer**
# 
# To select the IOB tag of each token while taking into account also the choices for other tokens. 
# 
# [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable) provides an implementation of CRF in PyTorch.
# 
# Init: `crf = torchcrf.CRF(num_tags, batch_first)`
# 
# Forward: `log_likelihood = crf(emissions, tags, mask)`
# 
# where `num_tags` the number of tags (classes), `emissions` the tags (classification) scores , `mask` a masking tensor to indicate the non padding tokens in the sequence and `tags` the tags labels.
# 
# The `forward` method return the `log_likelihood` score of each prediction, which can be summed or averaged over the batch or averaged by tokens.
# 
# Since we want to maximize the `log_likelihood`, in our optimization pipeline we will minimize the `negative_log_likelihood`, since out `optimizer` minimizes the `loss_function`.
# 
# Read more [here](https://en.wikipedia.org/wiki/Conditional_random_field)

# ## Transformers Sample Wrapper

# 
# For the second method we are going to use `Transformers` models, in particular we will adopt and finetuning in different tasks the [DistilBERT Encoder](https://huggingface.co/transformers/model_doc/distilbert.html) from [HuggingFace](https://huggingface.co/) (more about `DistilBERT` later).
# 
# We can test the contribution of using an attention based encoder in several tasks!
# 
# The bare `DistilBERT` encoder/transformer outputting raw hidden-states without any specific head on top. 
# 
# To load all the models related we will need to pass it's name to `from_pretrained` functions of different classes e.g. `AutoTokenizer`, `AutoConfig`, `AutoModel`, .. , which are needed to facilitate the use of transformers, we will discuss about it later.
# 
# Let's clarify what `distilbert-base-uncased` means:
# 
# * `distilbert` is the name of the model (a light version of `BERT` by `HuggingFace`)
# 
# * `base` means its standard version with 6 `Transformer` layers (`MultiHeadAttention` + +`LayerNorm` + `Linear`) 
# 
# * `uncased` means it makes no distintion between cased and uncased words, e.g. `HuggingFace` and `huggingface`
# 

# In[54]:


TRANSFORMER_NAME = 'distilbert-base-uncased'


# First of all let's redefine the `Sample` class in a more consistent way with respect to the `Transformer` models, in particular with respect to `DistilBERT`, which means each input should contains the following fields:
# 
# * `inputs_ids`: indices of input sequence tokens in the vocabulary.
# * `attention_mask`: Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1].
# 
# NOTE: `DistilBERT` do not require the `type_input_ids` information
# 
# The `Tokenizer` will make our life much easier, since it will produce these structures by itself in an automatic manner! In particular we will be using the `DistilBertTokenizerFast`.
# 
# In particular setting in the tokenizer `forward` method the values `return_offsets_mapping=True` it will return also the `offset_mapping` of the spliced words with respect to the sentence if `is_split_into_words=False` else with respect to each word if `is_split_into_words=True` (requires to split the sentence first). Which permits us to keep track of the splitting of the sentence and reconstruct it also in presence of special characters.

# In[55]:


tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME,use_fast=True) # equivalent to DistilBertTokenizerFast
# we can retreive the PAD_TOKEN_ID
PAD_TOKEN_ID = tokenizer.pad_token_id
# this token will identify the latter splitted parts of a word
# or the [CLS] and [SEP] tokens in the IOB labels
INV_TOKEN_ID = -1


# Let's define two new tokens which we will add to identify the query aspect term in the sentence:
# 
# NOTE: this part is intended for `Task B` but since we will be using the same `tokenizer` and `Sample` class definition, it is better to do it now.

# In[56]:


# needed for task B purposes
AS_TOKEN = '[AS]'
AE_TOKEN = '[AE]'
print('Vocabulary Size: ',len(tokenizer))
tokenizer.add_tokens([AS_TOKEN,AE_TOKEN], special_tokens=True)
print('New Vocabulary Size: ',len(tokenizer))


# Let's redefine the `Sample` class using the `AutoTokenizer` we have just defined, namely it will load the `DistilBertTokenizerFast` since we used the true flag `use_fast`:
# 
# NOTE: note that there are some methods required by the subsequent tasks e.g. `process_b()`, .. which will not be used here but later in the notebook.

# In[57]:


@dataclass
class Sample(ABSASample):
    text: str
    targets: Optional[List[Union[Tuple[int,int],str]]] = None
    categories: Optional[List[Tuple[str,str]]] = None

    def __post_init__(self):
        """These methods are called as soon as the dataclass is built"""
        self.order_targets()
        self.non_breaking_white_space_check()

    @cache
    def process(self) -> List[str]:
        """Tokenize the text"""
        encoding = tokenizer(self.text,is_split_into_words=False,            return_offsets_mapping=True,return_tensors='pt')
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['offset_mapping_word_piece'] = Sample.sentence2wordoffset(encoding['offset_mapping'])
        encoding['text'] = self.text
        return encoding

    @cache
    def process_a(self) -> Dict[str,torch.Tensor]:
        """Tokenize the text and produce IOB labels"""
        encoding = self.process()
        offset = encoding['offset_mapping']
        terms_indices, *_ = self.polarity_pairs()
        iob = Sample.termsindices2iob(terms_indices,offset)
        encoding['labels'] = iob
        return encoding
    
    @cache
    def process_b(self) -> Dict[str,torch.Tensor]:
        """Numericalize the sample"""
        old_text, old_terms_indices = self.add_special_tokens()
        encoding = self.process_a()
        _, terms, polarities = self.polarity_pairs()
        encoding['iob'] = encoding['labels']
        encoding['labels'] = polarities
        encoding['targets_indices'] = Sample.iob2targetsindices(encoding['iob'],encoding['offset_mapping'])
        encoding['terms'] = terms
        encoding['text'] = old_text
        encoding['terms_indices'] = old_terms_indices
        return encoding

    @cache
    def process_c(self) -> Dict[str,torch.Tensor]:
        """Numericalize the sample"""
        encoding = self.process()
        categories, _ = self.category_pairs()
        encoding['labels'] = categories
        return encoding
    
    @cache
    def process_d(self) -> Dict[str,torch.Tensor]:
        """Numericalize the sample"""
        encoding = self.process()
        categories, polarities = self.category_pairs()
        encoding['categories'] = categories
        encoding['labels'] = polarities
        return encoding

    @cache
    def order_targets(self) -> None:
        """Order targets by index wrt text""" 
        if self.targets is not None:
            if len(self.targets)>0:
                self.targets.sort(key=lambda x: x[0][0])

    def polarity_pairs(self) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Return terms and respective polarities.
        During test time, if no targets are provided it will 
        create FAKE labels, which will be discarded during inference time
        """
        terms_indices, terms, polarities = [[PAD_TOKEN_ID,PAD_TOKEN_ID]], [], [POLARITY[PAD_TOKEN]]
        if self.targets is not None:
            if len(self.targets)>0:
                terms_indices, terms, *polarities = zip(*self.targets)
                if len(polarities)>0:
                    # [0] because star operator return tuple (polarity,)
                    polarities = [POLARITY[i] for i in polarities[0]]
                else: # no labels -> test time
                    polarities = [POLARITY[PAD_TOKEN]]
        return torch.LongTensor(terms_indices), terms, torch.LongTensor(polarities)
    
    def category_pairs(self) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Return categories and respective polarities.
        During test time, if no targets are provided it will 
        create FAKE labels, which will be discarded during inference time
        """
        categories, polarities = [CATEGORY[PAD_TOKEN]], [POLARITY[PAD_TOKEN]]
        if self.categories is not None:
            if len(self.categories)>0:
                categories, *polarities = zip(*self.categories)
                if len(polarities)>0: # processing test task d
                    # [0] because star operator return tuple (polarity,)
                    polarities = [POLARITY[i] for i in polarities[0]]
                else: # no labels -> test time
                    polarities = [POLARITY[PAD_TOKEN]]
                categories = [CATEGORY[i] for i in categories]
        return torch.LongTensor(categories), torch.LongTensor(polarities)
    
    def non_breaking_white_space_check(self):
        """
        This method removes special characters (unicode) in the training code
        which apparently are not supported by the transformers 
        tokenizer and treated as blank space causing characters offsets 
        inconsistencies since instead the dataset treat them as part of the words.
        Thus we remove them and decrease the subsequent terms indices by 1.
        """
        try:
            while True:
                if '\u00a0' in self.text:
                    index = self.text.index('\u00a0')
                elif '\xa0' in self.text:
                    index = self.text.index('\xa0')
                else:
                    break
                self.text = self.text[:index]+self.text[index+1:]
                terms_indices, *_ = self.polarity_pairs()
                if len(terms_indices)>0:
                    mask = terms_indices>index
                    terms_indices[mask] -= 1
                    for i in range(len(terms_indices)):
                        self.targets[i][0]=terms_indices[i].tolist()
        except:
            pass
    
    @cache
    def add_special_tokens(self) -> str:
        """
        Special tokens [AS] and [AE] are added 
        before and after the terms to let the 
        Transformer models contextualize the token 
        according to the polarity of the sentence
        with respect to the term in between
        Return: 
            old_text (string): original text without special tokens
            terms_indices (string): original terms indices offsetts
        """
        terms_indices, *_ = self.polarity_pairs()
        old_text = self.text
        new_terms_indices = []
        split_list = []
        temp = 0
        inc = 0
        for start,end in terms_indices.tolist():
            if end!=0:
                split_list.append(self.text[temp:start])
                split_list.append(AS_TOKEN+' ')
                inc += len(split_list[-1])
                new_terms_indices.append([start+inc,end+inc])
                split_list.append(self.text[start:end])
                split_list.append(' '+AE_TOKEN)
                inc += len(split_list[-1])
                temp = end
        split_list.append(self.text[temp:])
        self.text = "".join(split_list)
        for i in range(len(new_terms_indices)):
            self.targets[i][0]=new_terms_indices[i]
        return old_text, terms_indices

    @staticmethod
    def termsindices2iob(
        terms_indices: torch.Tensor,
        offset: torch.Tensor,
        ) -> torch.Tensor:
        """
        Utility function which converts query terms 
        character-level indices into IOB encoding
        """
        iob = [IOB['O']]*len(offset)
        terms_indices = terms_indices[terms_indices[:,1]!=PAD_TOKEN_ID]
        for start, end in terms_indices:
            # here we must put >= because there are some wrong labels
            # in the dataset e.g. 'uninstall\u00a0' or "chicken in curry sauc"
            # infact some words (3 in restaurant) are cutted before the end
            i = int(torch.nonzero(offset[:,0]>=start).flatten()[0])
            i = i if start!=0 else 1
            j = int(torch.nonzero(offset[:,1]>=end).flatten()[0])
            iob[i]=IOB['B']
            iob[i+1:j+1]=[IOB['I']]*(j-i)
        iob = torch.LongTensor(iob)
        # put -1 on word pieces e.g 'ed','##gy','design' -> 'B','-1','I'
        # offset is infact [0,2],[2,4],[0,6]
        offset_wp = Sample.sentence2wordoffset(offset)
        # let's set [CLS] and [SEP] to have the invalid value -1
        iob[(offset_wp[:,0]==0) & (offset_wp[:,1]==0)] = INV_TOKEN_ID

        return iob

    @staticmethod
    def sentence2wordoffset(
        offset: torch.Tensor,
        ) -> torch.Tensor:
        """
        Utility function which converts sentence level offsets
        to word level offsets
        """
        offset_wp = torch.zeros_like(offset)
        offset_wp[0,:] = offset[0,:]
        start_word = offset_wp[0,0]
        for i in range(1,len(offset)):
            if offset[i,0] != offset[i-1,1]:
                start_word = offset[i,0]
            offset_wp[i,:] = offset[i,:] - start_word
        return offset_wp

    @staticmethod
    def targetsindices2termsindices(
        target_indices: torch.Tensor,
        offset: torch.Tensor,
        ) -> torch.Tensor:
        """
        Utility function which converts sequence level indices
        into characteer level indices within the sentence
        """
        terms_indices = [[offset[s,0],offset[e,1]] for s,e in target_indices]
        return torch.LongTensor(terms_indices)

    @staticmethod
    def iob2targetsindices(
        iob: torch.Tensor,
        offset: torch.Tensor,
        ) -> torch.Tensor:
        """
        Utility function which converts IOB encoding 
        into query target sequence-level indices
        """
        offset_wp = Sample.sentence2wordoffset(offset)
        b_indices = torch.nonzero(iob==IOB['B']).flatten()
        targets_indices = []
        if len(b_indices)>0:
            for start in b_indices:
                end = start
                for i in range(start+1,len(iob)-1): 
                    # len(iob)-1 since the last token 
                    # is [SEP] which value is -1, thus don't
                    # take it into account
                    if offset_wp[i,0] == offset_wp[i-1,1]: # if both B or I are splitted
                        end=i
                    elif iob[i] == IOB['I']: # if word is part of aspect term
                        end=i
                    else: # any other cases exit
                        break
                targets_indices.append([start,end])
        else:
            targets_indices = [[PAD_TOKEN_ID,PAD_TOKEN_ID]]
        return torch.LongTensor(targets_indices)
    
    @staticmethod
    def iob2termsindices(
        iob: torch.Tensor,
        offset: torch.Tensor,
        ) -> torch.Tensor:
        """
        Utility function which converts IOB encoding 
        into query terms character-level indices
        """
        targets_indices = Sample.iob2targetsindices(iob,offset)
        terms_indices = Sample.targetsindices2termsindices(targets_indices, offset)
        return terms_indices
    
    @staticmethod
    def termsindices2targetsindices(
        terms_indices: torch.Tensor,
        offset: torch.Tensor,
        )-> torch.Tensor:
        """
        Utility function which converts target sequence indices 
        into query terms character-level indices
        """
        iob = Sample.termsindices2iob(terms_indices,offset)
        targets_indices = Sample.iob2targetsindices(iob,offset)
        return targets_indices


# Here we define some more functions which let us better deal with decoding of the network results:

# In[58]:


""""
Task A utility functions
"""
def targetsindices2termsindices(
    batch_target_indices: torch.Tensor,
    batch_offset_mapping: torch.Tensor,
    ) -> torch.Tensor:
    """
    This function apply a conversion from indices at character level
    to word level, which means index with respect to the character sequence and 
    indices with respect to the tokenized sentence by using the offset_mapping
    Args:
        batch_target_indices (torch.Tensor): is a tensor containing the indices 
        with respect to the sequence length
        batch_offset_mapping (torch.Tensor): is a tensor containing the offset
        mapping sequence -> characters
    """
    terms_indices = []
    for targetsindices,off in zip(batch_target_indices,batch_offset_mapping):
        termsindices = Sample.targetsindices2termsindices(targetsindices,off)
        terms_indices.append(termsindices)
    terms_indices = torch.nn.utils.rnn.pad_sequence(terms_indices,        batch_first=True, padding_value=PAD_TOKEN_ID)
    return terms_indices

def iob2targetsindices(
    batch_iob_output: torch.Tensor,
    batch_offset_mapping: torch.Tensor,
    ) -> torch.Tensor:
    """
    This function retrtieves from IOB tagging the index of the targets 
    marked as 'B' and returns the index in which the target start and ends
    Args:
        batch_iob_output (torch.Tensor): is a tensor containing iob mapping
        of a sequence
    """
    targets_indices = []
    for iob,offset in zip(batch_iob_output,batch_offset_mapping):
        targetindices = Sample.iob2targetsindices(iob,offset)
        targets_indices.append(targetindices)
    targets_indices = torch.nn.utils.rnn.pad_sequence(targets_indices,        batch_first=True, padding_value=PAD_TOKEN_ID)
    return targets_indices

def iob2termsindices(
    batch_iob_output: torch.Tensor,
    batch_offset_mapping: torch.Tensor,
    ) -> torch.Tensor:
    """
    This function apply a conversion from iob tagging to char level indices, 
    which means index with respect to the character sequence by using the 
    offset_mapping
    Args:
        batch_iob_output (torch.Tensor): is a tensor containing iob mapping
        of a sequence
        batch_offset_mapping (torch.Tensor): is a tensor containing the offset
        mapping sequence -> characters
    """
    targets_indices = iob2targetsindices(batch_iob_output,batch_offset_mapping)
    terms_indices = targetsindices2termsindices(targets_indices,batch_offset_mapping)
    return terms_indices

def termsindices2terms(
    batch_terms_indices: List[List[Tuple[int,int]]],
    batch_texts: List[str],
    ) -> List[List[str]]:
    """
    This function translates terms indices to terms withing the respective sentence
    Args:
        batch_terms_indices (list): batch of terms indices
        batch_texts (list): batch of respective texts
    """
    terms_list = []
    for terms_indices, text in zip(batch_terms_indices,batch_texts):
        terms = [text[start:end] for start,end in terms_indices if end!=0]
        terms_list.append(terms)
    return terms_list

""""
Task C utility function
"""
def onehot2intlist(one_hot:torch.Tensor) -> List[int]:
    """Decodes onehot encoding returning 1s indices"""
    labels = []
    for elem in one_hot:
        elems = torch.nonzero(elem).tolist()
        labels.append(elems)
    return labels
    
""""
Task A+B+C+D utility functions
"""
def encode(
    batch: Union[List[Dict[str,str]],Dict[str,torch.Tensor]],
    ) -> List[Sample]:
    """Encode a List of dict into a List of Sample class"""
    return [Sample(**elem) for elem in batch]

def decode(
    batch: Union[List[Dict[str,str]],Dict[str,torch.Tensor]],
    task: str = 'a',
    ) -> List[Sample]:
    """
    This function execute different tasks according to the 
    task param passed, if can encode a dict into a Sample class or
    decode the predictions of the network depending on the task
    selected. 
    Args:
        batch (dict): is a dict containing all the relevant informations
        of the sequences
        task (str): can assume <a,b,c,d> values, which is the name of
        the operation to perform based on the SemEval 14 Task 4 task type
    """
    results = []
    if task == 'a':
        texts = batch['text']
        offset = batch['offset_mapping']
        terms_indices = iob2termsindices(batch['preds'], offset).tolist()
        terms_indices = [[[start,end] for start,end in term_indices if end!=0] for term_indices in terms_indices]
        terms = termsindices2terms(terms_indices,texts)
        # remove e.g. support. -> support 
        # since we are working with offsets most of the times punctuation after
        # words is also considered part of the word since there is no offset
        replacer = torchtext.data.functional.custom_replace([(r"[!?',;.:/<>`~=_+@#$%^&*()\[\]\{\}\|]", '')]) 
        terms = [replacer(term) for term in terms]
        targets = [list(map(list, zip(term_indices,term))) for term_indices,term in zip(terms_indices,terms)]
        results = [Sample(text=text,targets=target) for text,target in zip(texts,targets)]
    elif task == 'b':
        texts = batch['text']
        terms_indices = batch['terms_indices'].tolist()
        terms_indices = [[[start,end] for start,end in term_indices if end!=0] for term_indices in terms_indices]
        terms = batch['terms']
        polarities = [[POLARITY.itos[p] for p in pred if p!=PAD_TOKEN_ID] for pred in batch['preds'].tolist()]
        targets = [list(map(list, zip(term_indices,term,pol))) for term_indices,term,pol in zip(terms_indices,terms,polarities)]
        results = [Sample(text=text,targets=target) for text,target in zip(texts,targets)]
    elif task == 'c':
        texts = batch['text']
        batch_categories = [[[CATEGORY.itos[i] for i in p if i!=PAD_TOKEN_ID] for p in elem] for elem in onehot2intlist(batch['preds'].round())]
        results = [Sample(text=text,categories=categories) for text,categories in zip(texts,batch_categories)]
    elif task == 'd':
        texts = batch['text']
        batch_categories = [[CATEGORY.itos[l] for l in lab if l!=PAD_TOKEN_ID] for lab in batch['categories'].tolist()]
        polarities = [[POLARITY.itos[p] for p in elem] for elem in batch['preds'].tolist()]
        batch_categories = [list(zip(category,pol)) for category,pol in zip(batch_categories,polarities)]
        results = [Sample(text=text,categories=categories) for text,categories in zip(texts,batch_categories)]
    else:
        raise NotImplementedError
    
    return results


# ## Task A - Method 2

# ### Method 2 - DistilBERT+BiLSTM+CRF
# 
# For the second method we defined a different architecture.
# Starting from a general model based on [DistilBERT Encoder](https://huggingface.co/transformers/model_doc/distilbert.html) we stacker upon the other, the final model will contain:
# 
# `IOBDB`
# * `DistilBERT Encoder` 
# * `BiLSTM`
# * `Linear`
# * `Conditional Random Fields`
# 
# NOTE: loss functions, activations, batch normalizations and dropout are omitted, this is a general scheme.
# 
# We can test the contribution of using an `attention based encoder` in the IOB problem!
# 
# The `DistilBERT` model was proposed in the blog post `Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT`, and the paper `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter`. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.
# 
# Read more [here](https://huggingface.co/transformers/model_doc/distilbert.html)
# 
# The bare `DistilBERT` encoder/transformer outputting raw hidden-states without any specific head on top.
# 

# The `collate function` permits us to aggregate the samples in different ways according to the type of task.

# In[60]:


def collate_fn_a(
    samples: List[Sample], 
    device: Optional[torch.device] = None
    ) -> Dict[str,torch.Tensor]:
    """
    The collate function permits us to aggregate the samples in different 
    ways according to the type of task.
    Args:
        samples (list): list of Sample class.
    """
    pad = lambda x: torch.nn.utils.rnn.pad_sequence(x,batch_first=True, padding_value=PAD_TOKEN_ID)
    keys = samples[0].process_a().keys()
    values = zip(*[s.process_a().values() for s in samples])
    batch = {k: pad(v) if isinstance(v[0], torch.Tensor) else v for k,v in zip(keys,values)}
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)\
            else v for k, v in batch.items()}
    return batch


# Here we define a `custom LSTM layer` which performs pooling of the output; when indices are passed in the forward method it perform pooling at a the desired index of the sequence.

# In[61]:


class LSTM(nn.Module):
    def __init__(self, input_size, configs):
        super(LSTM, self).__init__()
        """
        This is a custom LSTM layer
        Args: 
            input_size (int): dimension of the last channel
            configs (Config): specs of the model
            Forward:
                sequence (torch.Tensor): [Batch x Time x Embs]
                the embedded input sequence
                lenghts (torch.Tensor): [Batch] the length of each 
                sequence, needed to specify where padding starts,
                it is used to avoid LSTM to process padding
                forward_index (torch.Tensor): [Batch] specify the 
                index where the forward pass should stop
                backward_index (torch.Tensor): [Batch] specify the
                index where the backward pass should stop
        Returns:
            sequence_output (torch.Tensor): [Batch x Time x Hidden]
            the output for each cell of the LSTM
            pooled_output (torch.Tensor): [Batch x Hidden] the pooled
            output of the LSTM corresponding to the indices passed 
            as input (forward_index, backward_index)
            last_hidden (torch.Tensor): [Batch x Hidden] hidden state
            of the last cell of the LSTM
        """
        self.configs = configs
        self.lstm = nn.LSTM(input_size, configs.hidden_dim,
                            bidirectional=configs.bidirectional,
                            num_layers=configs.num_layers, 
                            dropout = configs.dropout if \
                                configs.num_layers > 1 else 0,
                                batch_first = True)
        self.output_dim = configs.hidden_dim if configs.bidirectional             is False else configs.hidden_dim * 2
    def forward(
        self, 
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        forward_index: Optional[torch.Tensor]=None,
        backward_index: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        # lengths should be a mask of the type batch x words
        # which masks the padding (sequence!=PAD_TOKEN_ID).sum(-1) 
        batch, seq_length, embs = sequence.shape
        # remove padding
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,             lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(sequence)
        # re-add padding
        sequence_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,                 batch_first=True, padding_value=PAD_TOKEN_ID)
        batch, seq_length, embs = sequence_output.shape
        output = sequence_output.view(batch*seq_length,embs)
        # and we use a simple trick to compute a tensor of the indices
        # of the last token in each batch element
        # if no index is given we go forward till the end
        # while backward till the first token
        forward_index = lengths-1 if forward_index is None else forward_index
        backward_index = 0 if backward_index is None else backward_index
        offset_seq = torch.arange(batch, device=output.device) * seq_length
        forward_index = offset_seq + forward_index
        backward_index = offset_seq + backward_index
        if self.configs.bidirectional:
            # we retreive the output of the last token
            forward = output[forward_index,:embs//2] # batch_words x hidden_dim//2
            backward = output[backward_index,embs//2:] # batch_words x hidden_dim//2
            pooled_output = torch.cat([forward,backward],dim=-1)
        else:
            # we retreive the output of the last token
            pooled_output = output[forward_index] # batch_words x hidden_dim
            
        # last output (if bidir forward and backward concatenated)
        pooled_output = pooled_output.view(batch,embs)
        # the last hidden state [-1]
        last_hidden = h.view(self.configs.num_layers, -1,\
            self.output_dim)[-1] # batch x hidden_dim
        return sequence_output, pooled_output, last_hidden


# First of all, before passing to the model, we define a general function taken by `HuggingFace`'s `DistilBert` APIs needed to initialize the weights of the `head` of our network, namely the weights of the `nn.Linear`, `nn.Embedding` and `nn.LayerNorm` modules:

# In[62]:


"""
Copy paste code from https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html
"""
def init_weights(module):
    """Initialize the weights."""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02) # in DistilBertConfig std=0.02
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


# The model is based on a Transformer architecture and is composed by the following sublayers:
# 
# `IOBDB`
# * `DistilBERT Encoder` 
# * `BiLSTM`
# * `Linear`
# * `Conditional Random Fields`
# 
# Which I already introduced previously.

# In[63]:


class IOBDB(nn.Module):
    def __init__(self, configs):
        super(IOBDB, self).__init__()
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """       
        self.configs = configs
        if self.configs.bert_pretrained:
            self.DBERT = AutoModel.from_pretrained(self.configs.bert_name,\
                output_hidden_states=True)
        else:
            DBERTConfig = AutoConfig.from_pretrained(self.configs.bert_name,\
                output_hidden_states=True)
            self.DBERT = AutoModel.from_config(DBERTConfig)
            
        self.DBERT.resize_token_embeddings(self.configs.vocab_size)
        if not self.configs.bert_finetuning:
            for param in self.DBERT.parameters():
                param.requires_grad = False
        self.lstm = LSTM(self.DBERT.config.hidden_size, configs)
        output_dim = self.lstm.output_dim
        self.dropout = nn.Dropout(self.configs.dropout)
        self.layernorm = nn.LayerNorm(output_dim)
        self.classifier = nn.Linear(output_dim,\
            self.configs.num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)
        if self.configs.crf:
            # pytorch-crf computes internally the log likelihood
            self.crf = CRF(self.configs.num_classes,batch_first=True)
        else:
            # `ignore_index` parameter given to the loss function! It tells the loss
            # to ignore the predictions for those tokens labeled with `<pad>` 
            # to discard those predictions from the loss computation and the network
            # is not optimised to get predictions right for padding as well
            self.loss_function = nn.CrossEntropyLoss(ignore_index=self.configs.ignore_index)
        
        [init_weights(module) for module in [self.classifier,self.layernorm]];

    # This performs a forward pass of the model, as well as 
    # returning the predicted index.
    def forward(
        self, 
        batch: Dict[str,torch.Tensor]
        ) -> torch.Tensor:
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']}
        sequence = self.DBERT(**inputs).last_hidden_state
        lengths = batch['attention_mask'].sum(-1) # sequence length
        sequence = self.lstm(sequence, lengths)[0]
        sequence = self.layernorm(sequence)
        sequence = self.dropout(sequence)
        logits = self.classifier(sequence)
        results = batch
        offset = batch['offset_mapping_word_piece']
        # mask which set to 1 all tokens except [PAD], [CLS], [SEP]
        # in fact for these tokens offset is [0,0]
        # different from batch['attention_mask']
        mask = ~((offset[:,:,0]==self.configs.pad_token_id) &\
            (offset[:,:,1]==self.configs.pad_token_id))
        results['valids'] = mask
        results['logits'] = logits
        probs = self.softmax(logits)
        results['probs'] = probs
        # Compute the loss:
        if 'labels' in batch.keys():
            labels = batch['labels']
            # set to 0 the -1 ids (to be ignored)
            labels[labels==INV_TOKEN_ID] = self.configs.pad_token_id
            results['labels'] = labels
            if self.configs.crf:
                # we mask values in order to not consider padding
                # CRF computes the log likelihood (max problem),
                # while we want the negative log likelihood (min problem)
                # thus we add a negative sign
                # CRF requires first token to be valid, so we skip
                # [CLS] token 
                logits_crf = logits[:,1:,:] # CRF: sequence cannot start with masked value
                labels_crf = labels[:,1:]
                mask_crf = mask[:,1:]
                results['loss'] = - self.crf(logits_crf,labels_crf,
                    mask=mask_crf, reduction='token_mean')
            else:
                # We adapt the logits and labels to fit the format required 
                # for the loss function
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                results['loss'] = self.loss_function(logits, labels)

        if self.configs.crf:
            # CRF.decode returns a List[List[int]]
            # thus we need to pad it again to seq_length
            logits_crf = logits[:,1:,:] # CRF: sequence cannot start with masked value
            mask_crf = mask[:,1:]
            seq_length = logits.shape[1]-1
            predictions = self.crf.decode(logits_crf,mask=mask_crf)
            predictions[0] = predictions[0]+[self.configs.pad_token_id]*\
                (seq_length-len(predictions[0]))
            predictions = [torch.LongTensor(p) for p in predictions]
            predictions = torch.nn.utils.rnn.pad_sequence(predictions,
                batch_first=True, padding_value=self.configs.pad_token_id)
            predictions = torch.nn.functional.pad(input=predictions,\
                pad=(1, 0, 0, 0), mode='constant',\
                    value=self.configs.pad_token_id).to(logits.device)
        else:
            predictions = torch.argmax(probs, -1)
        
        predictions[~mask] = self.configs.pad_token_id
        results['preds'] = predictions
        
        if 'labels' in batch.keys(): # flatten for torchmetrics
            results['preds'] = results['preds'].view(-1)
            results['labels'] = results['labels'].view(-1)
        
        return results

# ## Task B

# ### DistilBERT+Linear
# 
# We need to identify the sentiment polarity of a terms in a sentence.
# The polarities for the terms and the categories are:
# * `positive`
# * `negative`
# * `neutral`
# * `conflict`
# 
# To perform term based sentiment analysis we need attention, thus we will use again a `Transformer` architecture, `DistilBERT`, in order to retreive polarity with respect to the key tokens. At this point given a sample like this one:
# <br>
# <br>
# {<br>
#   &emsp;  "text": "I love their pasta but I hate their Ananas Pizza.", <br>
#    &emsp; "targets": \[<br>
#    &emsp;&emsp;&emsp;         \[13, 17\], "pasta"], <br>
#     &emsp;&emsp;&emsp;        \[36, 47\], "Ananas Pizza"]<br>
#     &emsp;    \], <br>
# }
# <br>
# <br>
# where "pasta" -> "positive" and "Ananas Pizza" -> "negative".
# <br>
# <br>

# For the second task we defined a different architecture.
# Starting from a general model based on [DistilBERT Encoder](https://huggingface.co/transformers/model_doc/distilbert.html) we stacker upon the other, the final model will contain:
# 
# `POLDB`
# * `DistilBERT Encoder` 
# * `Linear`
# * `Linear`
# 
# We can test the contribution of using an attention based encoder in the POLARITY problem!
# 
# <!-- The `POLDB` model aims at producing `contextualized` words into the sentence thanks to attention, the `context` around the `targets` is the `POOLED` by a `BiLSTM` which `output` in `forward` and `backward` directions the states at `target_index-1` and `target_index+1`. The extracted information is then fed to a classification layer (actually 2 linear layers) which produces the probabilities over the 4 classes thanks to a `softmax` activation function. Of course a `CrossEntropy` loss function is used to `minimize the negative log likelihood estimation` over the `class distributions`. -->
# 
# The `POLDB` model aims at producing `contextualized` words into the sentence thanks to attention, by learning the contextualized representation of two new special tokens, `[AS]` and `[AE]`, which surround the `aspect terms`; the `contextualized embeddings of the special tokens` around the `targets` are then `concatenated` (positions `target_index-1` and `target_index+1`) and fed to a classification layer (actually 2 linear layers) which produces the probabilities over the 4 classes thanks to a `softmax` activation function. Of course a `CrossEntropy` loss function is used to `minimize the negative log likelihood estimation` over the `class distributions
# 
# There are also different ways to perform `pooling`, e.g. using one more attention module over the Transformer hidden states, but here I have found this strategy to be more effective.
# 
# `NOTE`: this strategy seems to be already present in the literature, namely here `wu2020transformerbased`, please notice also that they do not consider to `concatenate` the representation of the `[AE]` token to the `[AS]` one, which here resulted in giving higher performances.
# 
#  <font color='orange'>
# @misc{wu2020transformerbased,
# <br>
#  &emsp;title={Transformer-based Multi-Aspect Modeling for Multi-Aspect Multi-Sentiment Analysis}, 
#       <br>
#  &emsp;author={Zhen Wu and Chengcan Ying and Xinyu Dai and Shujian Huang and Jiajun Chen},
#       <br>
#  &emsp;year={2020},
#       <br>
#  &emsp;eprint={2011.00476},
#       <br>
#  &emsp;archivePrefix={arXiv},
#       <br>
#  &emsp;primaryClass={cs.CL}<br>
# }
# </font>

# NOTE: the `Sample` class we previously defined prepares also the data for task `b`!
# 

# Now, we need a new collate function for polarity purposes, which this time will call `sample.process_b()`:

# In[73]:


def collate_fn_b(
    samples: List[Sample], 
    device: Optional[torch.device] = None
    ) -> Dict[str,torch.Tensor]:
    """
    The collate function permits us to aggregate the samples in different 
    ways according to the type of task.
    Args:
        samples (list): list of Sample class.
    """
    pad = lambda x: torch.nn.utils.rnn.pad_sequence(x,batch_first=True, padding_value=PAD_TOKEN_ID)
    keys = samples[0].process_b().keys()
    values = zip(*[s.process_b().values() for s in samples])
    batch = {k: pad(v) if isinstance(v[0], torch.Tensor) else v for k,v in zip(keys,values)}
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)\
            else v for k, v in batch.items()}
    return batch


# In[74]:


class POLDB(nn.Module):
    def __init__(self, configs):
        super(POLDB, self).__init__()
        """
        Args:
            configs: (Config) a struct which contains all the 
            hyperparameters and configurations of the model
        """
        self.configs = configs
        if self.configs.bert_pretrained:
            self.DBERT = AutoModel.from_pretrained(self.configs.bert_name,                output_hidden_states=True)
        else:
            DBERTConfig = AutoConfig.from_pretrained(self.configs.bert_name,                output_hidden_states=True)
            self.DBERT = AutoModel.from_config(DBERTConfig)
            
        self.DBERT.resize_token_embeddings(self.configs.vocab_size)
        if not self.configs.bert_finetuning:
            for param in self.DBERT.parameters():
                param.requires_grad = False

        # self.lstm = LSTM(self.DBERT.config.hidden_size, configs)
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        hidden_size = 2*self.DBERT.config.hidden_size
        self.pre_classifier = nn.Linear(hidden_size, hidden_size,bias=False)
        self.dropout = nn.Dropout(self.configs.dropout)
        self.pre_layernorm = nn.LayerNorm(hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(hidden_size, self.configs.num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)
        # `ignore_index` parameter given to the loss function! It tells the loss
        # to ignore the predictions for those tokens labeled with `<pad>` 
        # to discard those predictions from the loss computation and the network
        # is not optimised to get predictions right for padding as well
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.configs.ignore_index) #weight=weights
        
        [init_weights(module) for module in [self.pre_classifier,\
            self.pre_layernorm,self.classifier,self.layernorm]];
    
    # This performs a forward pass of the model, as well as 
    # returning the predicted index.
    def forward(
        self, 
        batch: Dict[str,torch.Tensor]
        ) -> torch.Tensor:
        results = batch
        target_indices = batch['targets_indices']
        b, n, d = target_indices.shape
        valid = target_indices.sum(-1)>0
        valid_mask = valid.view(b*n)
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']}
        sequence = self.DBERT(**inputs).last_hidden_state
        # select the [AS] and [ES] tokens which are 1 step before and 
        # after the query token, if pad then do not increment
        aspect_start_index =  target_indices[...,0] - (target_indices[...,0]>0).long()
        aspect_end_index = target_indices[...,1] + (target_indices[...,1]>0).long()
        bs, ss, es = sequence.shape
        offset_seq = torch.arange(bs, device=sequence.device) * ss
        aspect_start_index = (offset_seq[...,None] + aspect_start_index).view(-1)
        aspect_end_index = (offset_seq[...,None] + aspect_end_index).view(-1)
        sequence = sequence.view(bs*ss,es)
        aspect_start = sequence[aspect_start_index] # batch_words x hidden_dim
        aspect_end = sequence[aspect_end_index] # batch_words x hidden_dim
        sequence = torch.cat([aspect_start,aspect_end],dim=-1)
        sequence = sequence.view(bs,n,2*es).contiguous()
        sequence = self.pre_layernorm(sequence)
        sequence = self.dropout(sequence)
        sequence = self.pre_classifier(sequence)
        sequence = self.gelu(sequence)
        sequence = self.layernorm(sequence)
        sequence = self.dropout(sequence)
        logits = self.classifier(sequence)
        # Compute the loss:
        # We adapt the logits and labels to fit the format required 
        # for the loss function
        logits[~valid] = float("-inf")
        if 'labels' in batch.keys():
            logits_ = logits.contiguous().view(-1, logits.shape[-1])[valid_mask]
            labels_ = batch['labels'].view(-1)[valid_mask]
            results['loss'] = self.loss_function(logits_, labels_)
        probs = self.softmax(logits)
        results['logits'] = logits
        results['probs'] = probs
        results['preds'] = torch.argmax(probs, -1)

        if 'labels' in batch.keys(): # flatten for torchmetrics
            results['preds'] = results['preds'].view(-1)
            results['labels'] = results['labels'].view(-1)
        
        return results

# ## Task C

# ### DistilBERT+Linear
# 
# We need to identify the category of the sentence. This time the task is a `MULTICLASS MULTILABELS CLASSIFICATION`, we will use a `BinaryCrossEntropyLoss`, namely `nn.BCELossWithLogits()`, since each sample can be one or more classes jointly. 
# 
# The possible categories are:
# * `anecdotes/miscellaneous`
# * `price`
# * `food`
# * `ambience`
# * `service`
# 
# To perform category based sentiment analysis we need attention, thus we will use again a `Transformer` architecture, `DistilBERT`. At this point given a sample like this one:
# <br>
# <br>
# {<br>
# &emsp;"categories": [<br>
# &emsp;&emsp;[<br>
# &emsp;&emsp;&emsp;"anecdotes/miscellaneous",<br>
# &emsp;&emsp;&emsp;"positive"<br>
# &emsp;&emsp;]<br>
# &emsp;],<br>
# &emsp;"targets": [],<br>
# &emsp;"text": "One of the more authentic Shanghainese restaurants in the US definitely the best in Manhattan Chinatown."<br>
# }
# <br>
# <br>
# 
# The model should predict:
# <br>
# <br>
# {<br>
# &emsp;"categories": [<br>
# &emsp;&emsp;[<br>
# &emsp;&emsp;&emsp;"anecdotes/miscellaneous",<br>
# &emsp;&emsp;]<br>
# &emsp;]<br>
# }
# <br>
# <br>
# 

# For the second task we defined a different architecture.
# Starting from a general model based on [DistilBERT Encoder](https://huggingface.co/transformers/model_doc/distilbert.html) we stacker upon the other, the final model will contain:
# 
# `CATDB`
# * `DistilBERT Encoder` 
# * `Linear`
# * `Linear`
# 
# We can test the contribution of using an attention based encoder in the POLARITY problem!
# 
# <!-- The `POLDB` model aims at producing `contextualized` words into the sentence thanks to attention, the `context` around the `targets` is the `POOLED` by a `BiLSTM` which `output` in `forward` and `backward` directions the states at `target_index-1` and `target_index+1`. The extracted information is then fed to a classification layer (actually 2 linear layers) which produces the probabilities over the 4 classes thanks to a `softmax` activation function. Of course a `CrossEntropy` loss function is used to `minimize the negative log likelihood estimation` over the `class distributions`. -->
# <!-- 
# The `POLDB` model aims at producing `contextualized` words into the sentence thanks to attention, by learning the contextualized representation of two new special tokens, `[AS]` and `[AE]`, which surround the `aspect terms`; the `contextualized embeddings of the special tokens` around the `targets` are then `concatenated` (positions `target_index-1` and `target_index+1`) and fed to a classification layer (actually 2 linear layers) which produces the probabilities over the 4 classes thanks to a `softmax` activation function. Of course a `CrossEntropy` loss function is used to `minimize the negative log likelihood estimation` over the `class distributions
# 
# There are also different ways to perform `pooling`, e.g. using one more attention module over the Transformer hidden states, but here I have found this strategy to be more effective.
# 
# `NOTE`: this strategy seems to be already present in the literature, namely here `wu2020transformerbased`, please notice also that they do not consider to `concatenate` the representation of the `[AE]` token to the `[AS]` one, which here resulted in giving higher performances.
# 
#  <font color='orange'>
# @misc{wu2020transformerbased,
# <br>
#  &emsp;title={Transformer-based Multi-Aspect Modeling for Multi-Aspect Multi-Sentiment Analysis}, 
#       <br>
#  &emsp;author={Zhen Wu and Chengcan Ying and Xinyu Dai and Shujian Huang and Jiajun Chen},
#       <br>
#  &emsp;year={2020},
#       <br>
#  &emsp;eprint={2011.00476},
#       <br>
#  &emsp;archivePrefix={arXiv},
#       <br>
#  &emsp;primaryClass={cs.CL}<br>
# }
# </font> -->

# NOTE: the `Sample` class we previously defined prepares also the data for task `c`!
# 

# Thus we need a new collate function, which this time will call `sample.process_c()`:

# In[84]:


def collate_fn_c(
    samples: List[Sample], 
    device: Optional[torch.device] = None
    ) -> Dict[str,torch.Tensor]:
    """
    The collate function permits us to aggregate the samples in different 
    ways according to the type of task.
    Args:
        samples (list): list of Sample class.
    """
    pad = lambda x: torch.nn.utils.rnn.pad_sequence(x,batch_first=True, padding_value=PAD_TOKEN_ID)
    keys = list(samples[0].process_c().keys())
    # keys.append(list(samples[0].process_d().keys()))
    values = zip(*[s.process_c().values() for s in samples])
    # values.append(*[s.process_d().values() for s in samples])
    batch = {k: pad(v) if isinstance(v[0], torch.Tensor) else v for k,v in zip(keys,values)}
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)\
            else v for k, v in batch.items()}
    return batch


# Here we define the new model:

# In[85]:


class CATDB(nn.Module):
    def __init__(self, configs):
        super(CATDB, self).__init__()
        """
        Args:
            configs: (Config) a struct which contains all the 
            hyperparameters and configurations of the model
        """
        self.configs = configs
        if self.configs.bert_pretrained:
            self.DBERT = AutoModel.from_pretrained(self.configs.bert_name,\
                output_hidden_states=True)
        else:
            DBERTConfig = AutoConfig.from_pretrained(self.configs.bert_name,\
                output_hidden_states=True)
            self.DBERT = AutoModel.from_config(DBERTConfig)
            
        self.DBERT.resize_token_embeddings(self.configs.vocab_size)
        if not self.configs.bert_finetuning:
            for param in self.DBERT.parameters():
                param.requires_grad = False
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        hidden_size = self.DBERT.config.hidden_size
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.configs.dropout)
        self.pre_layernorm = nn.LayerNorm(hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(hidden_size, self.configs.num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        # `ignore_index` parameter given to the loss function! It tells the loss
        # to ignore the predictions for those tokens labeled with `<pad>` 
        # to discard those predictions from the loss computation and the network
        # is not optimised to get predictions right for padding as well

        # This time we use BinaryCrossEntropyLoss, since we want to perform
        # MULTICLASS MULTILABELS CLASSIFICATION
        self.loss_function = nn.BCEWithLogitsLoss()
        [init_weights(module) for module in [self.pre_classifier,\
            self.pre_layernorm,self.classifier,self.layernorm]];
    
    # This performs a forward pass of the model, as well as 
    # returning the predicted index.
    def forward(
        self, 
        batch: Dict[str,torch.Tensor]
        ) -> torch.Tensor:
        results = batch
        inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']}
        sequence = self.DBERT(**inputs).last_hidden_state[:,0,:]
        # select the [CLS] token which represent the class of the text
        # after the query token, if pad then do not increment
        sequence = self.pre_layernorm(sequence)
        sequence = self.dropout(sequence)
        sequence = self.pre_classifier(sequence)
        sequence = self.gelu(sequence)
        sequence = self.layernorm(sequence)
        sequence = self.dropout(sequence)
        logits = self.classifier(sequence)
        # Compute the loss:
        # We adapt the logits and labels to fit the format required 
        # for the loss function
        if 'labels' in batch.keys():
            labels_ = batch['labels']
            labels_ = torch.nn.functional.one_hot(labels_,\
                num_classes=self.configs.num_classes).sum(1)\
                    .clamp(0,1).float().contiguous()
            results['loss'] = self.loss_function(logits, labels_)
            batch['labels'] = labels_.long()
        probs = self.sigmoid(logits)
        results['logits'] = logits
        results['probs'] = probs 
        preds = probs.round() # batch x classes
        preds[:,self.configs.ignore_index] = 0 # set to zero padding column
        results['preds'] = preds
        return results

# ## Task D

# ### DistilBERT+Linear
# 
# We are given multiple categories for a sentence:
# * `anecdotes/miscellaneous`
# * `price`
# * `food`
# * `ambience`
# * `service`
# 
# We now want to identify the sentiment polarity of the category of the sentence.
# The polarities for the terms and the categories are:
# * `positive`
# * `negative`
# * `neutral`
# * `conflict`
# 
# At this point given a sample like this one:
# <br>
# <br>
# {<br>
# &emsp;"categories": [<br>
# &emsp;&emsp;[<br>
# &emsp;&emsp;&emsp;"anecdotes/miscellaneous",<br>
# &emsp;&emsp;]<br>
# &emsp;],<br>
# &emsp;"targets": [],<br>
# &emsp;"text": "One of the more authentic Shanghainese restaurants in the US definitely the best in Manhattan Chinatown."<br>
# }
# <br>
# <br>
# Which true label is "positive", the network should predict:
# <br>
# <br>
# {<br>
# &emsp;"categories": [<br>
# &emsp;&emsp;[<br>
# &emsp;&emsp;&emsp;"anecdotes/miscellaneous",<br>
# &emsp;&emsp;&emsp;"positive"<br>
# &emsp;&emsp;]<br>
# &emsp;],<br>
# }
# <br>
# <br>

# For the second task we defined a different architecture.
# Starting from a general model based on [DistilBERT Encoder](https://huggingface.co/transformers/model_doc/distilbert.html) we stacker upon the other, the final model will contain:
# 
# `CATPOLDB`
# * `DistilBERT Encoder` 
# * `nn.ModuleList` x 5 (number of categories)
#      * `Linear`
#      * `Linear`
# 
# NOTE: activations, batch normalizations and dropout are omitted.
# 
# We can test the contribution of using an attention based encoder in the POLARITY problem!
# 
# <!-- The `POLDB` model aims at producing `contextualized` words into the sentence thanks to attention, the `context` around the `targets` is the `POOLED` by a `BiLSTM` which `output` in `forward` and `backward` directions the states at `target_index-1` and `target_index+1`. The extracted information is then fed to a classification layer (actually 2 linear layers) which produces the probabilities over the 4 classes thanks to a `softmax` activation function. Of course a `CrossEntropy` loss function is used to `minimize the negative log likelihood estimation` over the `class distributions`. -->
# <!-- 
# The `POLDB` model aims at producing `contextualized` words into the sentence thanks to attention, by learning the contextualized representation of two new special tokens, `[AS]` and `[AE]`, which surround the `aspect terms`; the `contextualized embeddings of the special tokens` around the `targets` are then `concatenated` (positions `target_index-1` and `target_index+1`) and fed to a classification layer (actually 2 linear layers) which produces the probabilities over the 4 classes thanks to a `softmax` activation function. Of course a `CrossEntropy` loss function is used to `minimize the negative log likelihood estimation` over the `class distributions
# 
# There are also different ways to perform `pooling`, e.g. using one more attention module over the Transformer hidden states, but here I have found this strategy to be more effective.
# 
# `NOTE`: this strategy seems to be already present in the literature, namely here `wu2020transformerbased`, please notice also that they do not consider to `concatenate` the representation of the `[AE]` token to the `[AS]` one, which here resulted in giving higher performances.
# 
#  <font color='orange'>
# @misc{wu2020transformerbased,
# <br>
#  &emsp;title={Transformer-based Multi-Aspect Modeling for Multi-Aspect Multi-Sentiment Analysis}, 
#       <br>
#  &emsp;author={Zhen Wu and Chengcan Ying and Xinyu Dai and Shujian Huang and Jiajun Chen},
#       <br>
#  &emsp;year={2020},
#       <br>
#  &emsp;eprint={2011.00476},
#       <br>
#  &emsp;archivePrefix={arXiv},
#       <br>
#  &emsp;primaryClass={cs.CL}<br>
# }
# </font> -->

# NOTE: the `Sample` class we previously defined prepares also the data for task `d`!
# 

# Let's load again the dataset, this time discarding all the samples which have no `targets` simply by passing `task='cd'` to `load_list()`:

# Thus we need a new collate function, which this time will call `sample.process_c()`:

# In[95]:


def collate_fn_d(
    samples: List[Sample], 
    device: Optional[torch.device] = None
    ) -> Dict[str,torch.Tensor]:
    """
    The collate function permits us to aggregate the samples in different 
    ways according to the type of task.
    Args:
        samples (list): list of Sample class.
    """
    pad = lambda x: torch.nn.utils.rnn.pad_sequence(x,batch_first=True, padding_value=PAD_TOKEN_ID)
    keys = list(samples[0].process_d().keys())
    values = zip(*[s.process_d().values() for s in samples])
    batch = {k: pad(v) if isinstance(v[0], torch.Tensor) else v for k,v in zip(keys,values)}
    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor)\
            else v for k, v in batch.items()}
    return batch


# Here we define the new model:

# In[96]:


class Parallel(nn.Module):
    """This module implements forward pass and stack over a nn.ModuleList"""
    def __init__(self, module_list: nn.ModuleList):
        super().__init__()
        self.module_list = module_list

    def forward(self, inputs):
        return torch.stack([module(inputs) for module in self.module_list],dim=1)


# In[97]:


class CATPOLDB(nn.Module):
    def __init__(self, configs):
        super(CATPOLDB, self).__init__()
        """
        Args:
            configs: (Config) a struct which contains all the 
            hyperparameters and configurations of the model
        """
        self.configs = configs
        if self.configs.bert_pretrained:
            self.DBERT = AutoModel.from_pretrained(self.configs.bert_name,\
                output_hidden_states=True)
        else:
            DBERTConfig = AutoConfig.from_pretrained(self.configs.bert_name,\
                output_hidden_states=True)
            self.DBERT = AutoModel.from_config(DBERTConfig)
            
        self.DBERT.resize_token_embeddings(self.configs.vocab_size)
        if not self.configs.bert_finetuning:
            for param in self.DBERT.parameters():
                param.requires_grad = False

        # self.lstm = LSTM(self.DBERT.config.hidden_size, configs)
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        hidden_size = self.DBERT.config.hidden_size
        from copy import deepcopy
        classifier_pipeline = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.configs.dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.configs.dropout),
            nn.Linear(hidden_size, self.configs.num_classes),
        )
        parallel_classifier_pipeline = nn.ModuleList([deepcopy(classifier_pipeline)]*\
            self.configs.num_categories)

        self.parallel_classifiers = Parallel(parallel_classifier_pipeline)
        self.softmax = torch.nn.Softmax(dim=-1)
        # `ignore_index` parameter given to the loss function! It tells the loss
        # to ignore the predictions for those tokens labeled with `<pad>` 
        # to discard those predictions from the loss computation and the network
        # is not optimised to get predictions right for padding as well
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.configs.ignore_index)
        #weight=weights
        [[init_weights(layer) for layer in module.modules()]\
            for module in self.parallel_classifiers.module_list];
    
    # This performs a forward pass of the model, as well as 
    # returning the predicted index.
    def forward(
        self, 
        batch: Dict[str,torch.Tensor]
        ) -> torch.Tensor:
        results = batch
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        sequence = self.DBERT(**inputs).last_hidden_state[:,0,:]
        # select the [CLS] token which represent the class of the text
        # after the query token, if pad then do not increment
        logits = self.parallel_classifiers.forward(sequence) # batch_size x categories x polarity
        b, nc, cl = logits.shape # batch_size x categories x polarity
        # Compute the loss:
        # We adapt the logits and labels to fit the format required 
        # for the loss function
        categories =  batch['categories']
        b, c = categories.shape
        batch_offset = torch.arange(b,device=sequence.device) * nc
        categories_indices = (batch_offset[:,None] + categories).view(b * c)
        logits = (logits.view(b*nc,cl)[categories_indices]).view(b, c, cl).contiguous()
        if 'labels' in batch.keys():
            logits_ = logits.view(-1,logits.shape[-1])
            labels_ = batch['labels'].view(-1)
            results['loss'] = self.loss_function(logits_, labels_)
        probs = self.softmax(logits)
        results['logits'] = logits
        results['probs'] = probs
        results['preds'] = torch.argmax(probs, -1)

        if 'labels' in batch.keys(): # flatten for torchmetrics
            mask = results['labels'].view(-1)!=self.configs.pad_token_id
            results['preds'] = results['preds'].view(-1)[mask]
            results['labels'] = results['labels'].view(-1)[mask]
        
        return results

# ## Task Wrapper B/A+B/C+D

# Here we define all the configurations files:

# In[104]:


# we define all relevant configs for the model

configs_iob = Config(**{
    'model': IOBDB,
    'vocab_size': len(tokenizer),
    'num_classes': len(IOB), 
    'dropout': 0.5,
    'crf': True,
    'lr': 2e-5,
    'ignore_index': PAD_TOKEN_ID,
    'hidden_dim': 348,
    'bidirectional': True,
    'num_layers': 2,
    'bert_name': TRANSFORMER_NAME,
    'bert_pretrained': False,
    'bert_finetuning': True,
    'pad_token_id': PAD_TOKEN_ID,
    'pre_trained_path':  './model/a2-epoch=16-val_F1=0.881.ckpt',
    })

configs_pol = Config(**{
    'model': POLDB,
    'vocab_size': len(tokenizer),
    'num_classes': len(POLARITY), 
    'dropout': 0.3,
    'lr': 2e-5,
    'ignore_index': POLARITY[PAD_TOKEN],
    'bert_name': TRANSFORMER_NAME,
    'bert_pretrained': False,
    'bert_finetuning': True,
    'pad_token_id': PAD_TOKEN_ID,
    'pre_trained_path':  './model/b-epoch=15-val_F1=0.601.ckpt',
    })

configs_cat = Config(**{
    'model': CATDB,
    'vocab_size': len(tokenizer),
    'num_classes': len(CATEGORY), 
    'dropout': 0.4,
    'lr': 5e-5,
    'ignore_index': CATEGORY[PAD_TOKEN],
    'bert_name': TRANSFORMER_NAME,
    'bert_pretrained': False,
    'bert_finetuning': True,
    'pad_token_id': PAD_TOKEN_ID,
    'pre_trained_path':  './model/c-epoch=11-val_F1=0.855.ckpt',

    })

configs_catpol = Config(**{
    'model': CATPOLDB,
    'vocab_size': len(tokenizer),
    'num_classes': len(POLARITY), 
    'num_categories': len(CATEGORY), 
    'dropout': 0.4,
    'lr': 5e-5,
    'ignore_index': POLARITY[PAD_TOKEN],
    'bert_name': TRANSFORMER_NAME,
    'bert_pretrained': False,
    'bert_finetuning': True,
    'pad_token_id': PAD_TOKEN_ID,
    'pre_trained_path':  './model/d-epoch=08-val_F1=0.614.ckpt',
    })


# 
# The whole model is the conjunction of the four models (with some `post-processing`). We can use the configuration files defined before and build the final model!

# In[105]:


class ABSADB(nn.Module):
    def __init__(
        self, 
        mode = 'b',
        device: Optional[torch.device] = torch.device('cpu'),
        ) -> None:
        super(ABSADB, self).__init__()
        """
        Args:
            mode (str): specifies the type of task
            device (torch.device): indicates the current
            device on which building the model
        """
        if mode not in ['b','ab','cd']:
            raise NotImplementedError

        self.device = device
        self.mode = mode
        
        if self.mode=='ab':
            self.iob_model = ABSAModule(configs_iob) # subtask a
            self.iob_model.load_state_dict(
                torch.load(configs_iob.pre_trained_path,\
                map_location=self.device))
            self.iob_model.to(self.device)
        if self.mode=='b' or self.mode=='ab':
            self.pol_model = ABSAModule(configs_pol) # subtask b
            self.pol_model.load_state_dict(
                torch.load(configs_pol.pre_trained_path,\
                map_location=self.device))
            self.pol_model.to(self.device)
        if self.mode=='cd':
            self.cat_model = ABSAModule(configs_cat) # subtask c
            self.cat_model.load_state_dict(
                torch.load(configs_cat.pre_trained_path,\
                map_location=self.device))
            self.cat_model.to(self.device)

            self.catpol_model = ABSAModule(configs_catpol) # subtask d
            self.catpol_model.load_state_dict(
                torch.load(configs_catpol.pre_trained_path,\
                map_location=self.device))
            self.catpol_model.to(self.device)

    # This performs a forward pass of the model, as well as 
    # returning the predictions.
    def forward(
        self, 
        batch: List[Dict[str,str]],
        return_dict: bool = True,
        ) -> torch.Tensor:
        batch = encode(batch)
        if self.mode == 'ab':
            batch = collate_fn_a(batch,device=self.device)
            iob_results = self.iob_model.predict(batch)
            iob_results = self.to_cpu(iob_results)
            batch = decode(iob_results,task='a')

        if self.mode=='b' or self.mode=='ab':
            batch = collate_fn_b(batch,device=self.device)
            pol_results = self.pol_model.predict(batch)
            pol_results = self.to_cpu(pol_results)
            batch = decode(pol_results,task='b')

        if self.mode=='cd':
            batch = collate_fn_c(batch,device=self.device)
            cat_results = self.cat_model.predict(batch)
            cat_results = self.to_cpu(cat_results)
            batch = decode(cat_results,task='c')

            batch = collate_fn_d(batch,device=self.device)
            catpol_results = self.catpol_model.predict(batch)
            catpol_results = self.to_cpu(catpol_results)
            batch = decode(catpol_results,task='d')

        if return_dict:
            if self.mode=='b' or self.mode=='ab':
                batch = [{'targets':[tuple(target[-2:])\
                    for target in sample.targets]} for sample in batch]
            if self.mode=='cd':
                batch = [{'categories':[tuple(category)\
                    for category in sample.categories]} for sample in batch]
        return batch

    def to_cpu(
        self,
        batch: Dict[str,torch.Tensor],
        ) -> Dict[str,torch.Tensor]:
        """Move to CPU in order to pre/post-process the input/output"""
        return {k: v.cpu() if isinstance(v,torch.Tensor)\
            else v for k,v in batch.items()}
