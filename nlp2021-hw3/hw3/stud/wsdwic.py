#!/usr/bin/env python
# coding: utf-8

from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy
from sparselinear import SparseLinear



from nltk.corpus import wordnet as wn

MAPPING = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

SYNSETS = list(wn.all_synsets())

SYNSETS_TO_IDS = { synset.name(): index for index, synset in enumerate(SYNSETS) }

IDS_TO_SYNSETS = { index: synset_name for synset_name, index in SYNSETS_TO_IDS.items() }

print('Number Synsets: ',len(SYNSETS))


def get_adjacency_matrix() -> torch.Tensor:

    """ This function return an adjacency matrix based onf Hypo+Hyper """

    indices = []
    values = []
    shape = (len(SYNSETS),len(SYNSETS))

    hypo = lambda s: s.hyponyms()
    hyper = lambda s: s.hypernyms()

    depth = 1

    for index, synset_name in IDS_TO_SYNSETS.items():
        linked_synsets = list(wn.synset(synset_name).closure(hypo, depth=depth)) + \
            list(wn.synset(synset_name).closure(hyper, depth=depth))
        factor = len(linked_synsets)
        for linked in linked_synsets:
            indices.append([index,SYNSETS_TO_IDS[linked.name()]])
            values.append(1/factor)

    ADJACENCY_MATRIX = torch.sparse_coo_tensor(list(zip(*indices)),\
        values,shape,dtype=torch.float32).coalesce()

    return ADJACENCY_MATRIX

def cache(method):
    """
    This decorator caches the return value of a method so that results are not recomputed
    """
    method_name = method.__name__
    def wrapper(self, *args, **kwargs):
        self._cache = getattr(self, '_cache', {})
        if method_name not in self._cache:
            self._cache[method_name] = method(self, *args, **kwargs)
        return self._cache[method_name]
    return wrapper

@dataclass
class Sample:
    """ Basic Implementation of a Sample Instance """
    id: str
    lemma: str
    pos: str
    sentence1: str
    sentence2: str
    start1: str
    end1: str
    start2: str
    end2: str
    label: Optional[str] = None

    @cache
    def process(self):
        """ Convert samples to ids """
        tokens_tuple = list([self.sentence1,self.sentence2])
        tokens_offsets = list([
            list([int(self.start1),int(self.end1)]),
            list([int(self.start2),int(self.end2)])
            ])
        synset_list = wn.synsets(self.sentence1[int(self.start1):int(self.end1)],\
            pos=MAPPING[self.pos])
        synset_list = synset_list if len(synset_list)>0 \
             else wn.synsets(self.sentence2[int(self.start2):int(self.end2)],\
                 pos=MAPPING[self.pos])
        synsets = [SYNSETS_TO_IDS[synset.name()] for synset in synset_list]
        return tokens_tuple, tokens_offsets, synsets
        
def to_sample(sentence_pairs:  List[Dict]) -> List[Sample]:
    from tqdm import tqdm
    samples = []
    for sample_dict in tqdm(sentence_pairs, desc="parsing samples: "):    
        samples.append(Sample(**sample_dict))
    print('\nDone.')
    return samples


def collate_fn(samples: List[Sample]) -> List[Dict]:
    tokens, offsets, synsets = zip(*[s.process() for s in samples])
    batch = dict()
    batch['tokens1'] = list([token[0] for token in tokens])
    batch['key_indices1'] = list([offset[0] for offset in offsets])
    batch['tokens2'] = list([token[1] for token in tokens])
    batch['key_indices2'] = list([offset[1] for offset in offsets])
    batch['synsets'] = list(synsets)
    
    return batch

import abc 

class BaseKeyEmbedder(torch.nn.Module, metaclass=abc.ABCMeta):

    embedding_dim: int
    n_hidden_states: int
    retrain_model: bool
    is_split_into_words: bool

    def __init__(self, retrain_model: bool = False):
        super().__init__()
        self.retrain_model = retrain_model

    def forward(
            self,
            key_indices: Union[List[int], List[List[int]]] = None, 
            src_tokens_str: Union[List[List[str]], List[str]] = None,
            batch_major: bool = True,
            **kwargs
    ):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    @property
    def embed_dim(self):
        return self.embedding_dim

    @property
    def embedded_dim(self):
        return self.embedding_dim

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)


from pprint import pprint
class BERTKeyEmbedder(BaseKeyEmbedder):

    DEFAULT_MODEL = 'bert-base-uncased'

    @staticmethod
    def _do_imports():
        import transformers as tnf
        import torchtext.data.functional as ttext
        return tnf, ttext

    def __init__(
        self,
        name: Union[str, None] = None,
        key_token: str = '*',
        weights: str = "",
        is_split_into_words: bool = True,
        key_piece_merging_mode: str = 'MEAN',
        last_hidden_number: int = 4,
        last_hidden_merging_mode: str = 'SUM',
        retrain_model: bool = False,
    ):
        
        assert not retrain_model
        super(BaseKeyEmbedder, self).__init__()

        assert not retrain_model
        assert last_hidden_number > 0 and last_hidden_number < 12

        self.retrain_model = retrain_model
        if not name:
            name = self.DEFAULT_MODEL

        self.name = name
        self.is_split_into_words = is_split_into_words
        self.last_hidden_number = last_hidden_number
        self.key_piece_merging_mode = key_piece_merging_mode
        self.last_hidden_merging_mode = last_hidden_merging_mode

        tnf, ttext = self._do_imports()
        self.bert_tokenizer = tnf.BertTokenizerFast.from_pretrained(name,do_lower_case=True)
        # just load model without downloading the parameters, will be
        # overwritten by the saved trained model
        config = tnf.AutoConfig.from_pretrained(name,output_hidden_states= True)
        self.bert_model = tnf.BertModel(config)
        self.hidden_size = self.bert_model.config.hidden_size

        self.key_token = key_token
        self.key_id = self.bert_tokenizer.convert_tokens_to_ids(self.key_token)
        self.cleaner = ttext.custom_replace([(r"" + "\\" + self.key_token, '')])

        for par in self.parameters():
            par.requires_grad = False

    def forward(
        self, 
        key_indices: Union[List[int], List[List[int]]], 
        src_tokens_str: Union[List[List[str]], List[str]], 
        batch_major=True,
        **kwargs
        )-> torch.Tensor:
        new_src_tokens_str = []

        if self.is_split_into_words:
            assert isinstance(key_indices[0], int)
            
            for src_token_str, key_index in zip(src_tokens_str,key_indices):
                new_src_token_str = list(self.cleaner(src_token_str))
                # surround the query token
                new_src_token_str[key_index] = self.key_token +\
                        new_src_token_str[key_index] + self.key_token
                new_src_tokens_str.append(new_src_token_str)
        else:
            assert isinstance(key_indices[0], list) and isinstance(key_indices[0][0], int)

            for src_token_str, key_index in zip(src_tokens_str,key_indices):
                new_src_token_str = list(self.cleaner([src_token_str]))[0]
                # surround the query token
                new_src_token_str = new_src_token_str[:key_index[0]] + self.key_token + \
                        new_src_token_str[key_index[0]:key_index[1]] + self.key_token + \
                        new_src_token_str[key_index[1]:]
                new_src_tokens_str.append(new_src_token_str)

        input_batch = self.bert_tokenizer(
                new_src_tokens_str,
                return_tensors='pt',
                is_split_into_words=self.is_split_into_words,
                padding=True
            )

        input_batch['query_token_indices'] = self.get_splitted_input_indices(input_batch['input_ids'])
        # input_batch['sequence_length'] = input_batch['attention_mask'].clone().detach().sum(dim=-1)

        with torch.set_grad_enabled(self.retrain_model and not self.training):
            outputs = self.bert_model.eval().forward(
                input_ids=input_batch['input_ids'].to(self.device),
                attention_mask=input_batch['attention_mask'].to(self.device),
                token_type_ids=input_batch['token_type_ids'].to(self.device),
            )
        
        hidden_states = outputs[2]

        stacked_hidden_states = torch.stack([hidden_states[i] for i in \
            torch.arange(start=-1,end=-(self.last_hidden_number+1),step=-1)], dim=-1)
        
        if self.last_hidden_merging_mode == 'SUM':
            merged_hidden_states = stacked_hidden_states.sum(dim=-1)
        elif self.last_hidden_merging_mode == 'MEAN':
            merged_hidden_states = stacked_hidden_states.mean(dim=-1)
        else:
            merged_hidden_states = stacked_hidden_states.mean(dim=-1)
        
        # contextualized words to disambiguate
        key_context_embedding = self.get_word(
                        merged_hidden_states,
                        input_batch['query_token_indices'][...,0].to(self.device),
                        input_batch['query_token_indices'][...,1].to(self.device)
                        )
        return key_context_embedding
    
    def get_splitted_input_indices(
        self,
        batch_input_ids: torch.Tensor,
        ) -> torch.Tensor:

        splitted_input_indices = []
        for input_ids in batch_input_ids:
            indices = (input_ids == self.key_id).nonzero(as_tuple=True)[0]
            indices[0]+=1
            splitted_input_indices.append(indices)
        tensor_indices = torch.stack(splitted_input_indices,dim=0)
        assert tensor_indices.shape[-1] == 2

        return tensor_indices

    def get_word(
        self,
        tensor: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor
        ) -> torch.Tensor:

        B,N,C = tensor.shape
        tensor = tensor.view(B*N,C)
        indices = torch.cat([b*N+torch.arange(start[b],end[b],1) for b in torch.arange(B)],dim=0)
        mask = torch.zeros_like(tensor)
        mask[indices,:]=1
        tensor = tensor * mask
        num_elem = (end-start)[...,None].repeat(1, C)
        tensor = tensor.view(B,N,C)

        if self.key_piece_merging_mode == 'SUM':
            tensor = tensor.sum(dim=1)
        elif self.key_piece_merging_mode == 'MEAN':
            tensor = tensor.mean(dim=1)
        else:
            tensor = tensor.mean(dim=1)

        tensor = tensor/num_elem

        return tensor   


class WSDModel(pl.LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        super(WSDModel, self).__init__(*args, **kwargs)
        self.save_hyperparameters(hparams.__dict__)

        self.bert_key_embedder = BERTKeyEmbedder(name = hparams.wsd_name,
            is_split_into_words=hparams.is_split_into_words)
        # assert Bert if freeze
        self.bert_key_embedder.training = False
        self.vocab_size = hparams.vocab_size
        
        self.adjacency = SparseLinear(hparams.vocab_size,hparams.vocab_size,\
            connectivity = hparams.coalesce_adjacency_matrix.indices(),bias=False)
        self.adjacency.weights = torch.nn.Parameter(
            hparams.coalesce_adjacency_matrix.values())
        self.adjacency.requires_grad = hparams.train_adjacency

        self.dropout = torch.nn.Dropout(hparams.dropout)
        self.bnorm = torch.nn.BatchNorm1d(self.bert_key_embedder.hidden_size)
        self.swish = torch.nn.SiLU()
        self.classifier1 = torch.nn.Linear(self.bert_key_embedder.hidden_size, hparams.hidden_size)
        self.classifier2 = torch.nn.Linear(hparams.hidden_size, hparams.vocab_size, bias=False)

        torch.nn.init.xavier_uniform_(self.classifier1.weight)
        torch.nn.init.xavier_uniform_(self.classifier2.weight)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.lr = hparams.lr

        # Measures
        metrics = MetricCollection([Accuracy(num_classes=hparams.vocab_size)])
        self.train_metrics = metrics.clone(prefix='wsd_train_')
        self.val_metrics = metrics.clone(prefix='wsd_val_')
        self.test_metrics =  metrics.clone(prefix='wsd_test_')

    def forward(
        self,
        batch: Union[Dict[str,List], torch.Tensor],
        ) -> torch.Tensor:
        """
        Computes the forward pass, returning unnormalized log probabilities (the logits)
        """
        if isinstance(batch, dict):
            embed = self.embed(batch)
            out = self.bnorm(embed)
            out = self.classifier1(out)
            out = self.swish(out)
        elif isinstance(batch, torch.Tensor):
            act = batch
        else:
            act = batch
        
        out = self.classifier2(act)
        logits = self.adjacency(out) + out

        return logits

    @torch.no_grad()
    def embed(
        self,
        batch: Dict[str,List]
        )-> torch.Tensor:

        tokens = batch['tokens']
        key_indices = batch['key_indices']
        embed = self.bert_key_embedder.forward(key_indices=key_indices,src_tokens_str=tokens)

        return embed

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str,List],
        ) -> Dict[str, torch.Tensor]:
        """
        Computes a batch of predictions (as list of int) from logits
        """
        embed = self.embed(batch)
        out = self.bnorm(embed)
        out = self.classifier1(out)
        act = self.swish(out)
        logits = self(act)
        mask = self.mask_synsets(batch['synsets']).to(self.device)
        pred = torch.argmax(self.softmax(logits)\
            .sparse_mask(mask).to_dense(),dim=-1)
        return {'pred': pred, 'act': act}


    def mask_synsets(
        self,
        batch_indices: List[List[int]],
        ) -> torch.Tensor:
        indices = []
        for row, elem in enumerate(batch_indices):
            for col in elem:
                indices.append([row,col])
        values = torch.ones((len(indices,)))
        sparse_mask = torch.sparse_coo_tensor(
            list(zip(*indices)),
            values,
            (len(batch_indices),self.vocab_size)
            )
        return sparse_mask.coalesce()

    def basic_step(
        self,
        batch: Dict[str,List],
        ) -> Dict[str,torch.Tensor]:
        """
        Evaluates performance on ground truth in terms of both loss (returned)
        and metrics are update
        """
        embed = self.embed(batch)
        out = self.bnorm(embed)
        out = self.classifier1(out)
        act = self.swish(out)
        logits = self(act)
        
        pred = torch.argmax(self.softmax(logits),dim=-1)
        gold = torch.tensor(batch['labels']).to(self.device)
        loss = self.loss(logits,gold)
        return {'loss': loss, 'pred': pred, 'gold': gold, 'act': act}

    def training_step(
        self,
        batch: Dict[str,List],
        batch_idx: int
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        Computes loss to be used for .backward()
        """
        result = self.basic_step(batch)
        return result

    def write_metrics_end(
        self,
        batch_parts: Dict[str,torch.Tensor],
        metrics: MetricCollection,
        ):
        """
        Write metrics at end on multi GPUs
        """
        output = metrics(batch_parts['pred'],batch_parts['gold'])
        self.log_dict(output, on_step=True, on_epoch=False, prog_bar=True)

    def training_step_end(
        self,
        batch_parts: Dict[str,torch.Tensor],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward() on multi GPUs
        """
        self.write_metrics_end(batch_parts,self.train_metrics)
        return batch_parts['loss']

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str,List],
        batch_idx: int
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        Evaluates on batch of validation samples
        """
        result = self.basic_step(batch)
        return result

    def validation_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward()
        """
        self.write_metrics_end(batch_parts,self.val_metrics)

    @torch.no_grad()
    def test_step(
        self,
        batch: Dict[str,List],
        batch_idx: int
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        Evaluates on batch of test samples
        """
        result = self.basic_step(batch)
        return result

    def test_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward()
        """
        self.write_metrics_end(batch_parts,self.test_metrics)

    def configure_optimizers(self):
        """
        [Required by lightning]
        Initializes the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lf)
        return optimizer

    def basic_on_epoch_end(
        self,
        metrics: MetricCollection,
        ):
        """
        Log reduction of metrics
        """
        output = metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        """
        [lightning]
        Logging and EM reset (validation)
        """
        self.basic_on_epoch_end(self.train_metrics)

    def on_validation_epoch_end(self):
        """
        [lightning]
        Logging and EM reset (validation)
        """
        self.basic_on_epoch_end(self.val_metrics)

    def on_test_epoch_end(self):
        """
        [lightning]
        Logging and EM reset (test)
        """
        self.basic_on_epoch_end(self.test_metrics)


class WiCWSDModel(pl.LightningModule):


    def __init__(self, hparams, device, *args, **kwargs):
        super(WiCWSDModel, self).__init__(*args, **kwargs)
        self.save_hyperparameters(hparams.__dict__)

        self.bert_wsd = WSDModel(hparams)

        if hparams.wsd_checkpoint != None:
            checkpoint = torch.load(hparams.wsd_checkpoint, map_location=device)
            self.bert_wsd.load_state_dict(checkpoint['state_dict'])
            # assert Bert if freeze
            for par in self.bert_wsd.parameters():
                par.requires_grad = False
            self.bert_wsd.eval()
            print('WSD Model loaded.')

        self.swish = torch.nn.SiLU()
        self.classifier1 = torch.nn.Linear(hparams.hidden_size, hparams.hidden_size//2)
        self.bnorm = torch.nn.BatchNorm1d(hparams.hidden_size//2)
        self.classifier2 = torch.nn.Linear(hparams.hidden_size, 1)
        self.dropout = torch.nn.Dropout(hparams.dropout)

        torch.nn.init.xavier_uniform_(self.classifier1.weight)
        torch.nn.init.xavier_uniform_(self.classifier2.weight)

        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
        self.lr = hparams.lr

        # Measures
        metrics = MetricCollection([Accuracy(num_classes=1)])
        self.train_metrics = metrics.clone(prefix='wic_train_')
        self.val_metrics = metrics.clone(prefix='wic_val_')
        self.test_metrics =  metrics.clone(prefix='wic_test_')

    def forward(
        self,
        batch: Tuple[torch.Tensor,torch.Tensor],
        ) -> torch.Tensor:
        """
        Computes the forward pass, returning unnormalized log probabilities (the logits)
        """
        outs = [self.swish(self.bnorm(self.classifier1(logit))) for logit in [batch[0],batch[1]]]
        logits = self.classifier2(self.dropout(torch.cat(outs,dim=-1))).squeeze(-1)

        return logits
    
    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str,List],
        ) -> Dict[str, List]:
        """
        Computes a batch of predictions (as list of int) from logits
        """
        pack1, pack2 = self.unpack_sentence_pairs(batch)
        result1 = self.bert_wsd.eval().predict(pack1)
        result2 = self.bert_wsd.eval().predict(pack2)
        logits = self((result1['act'], result2['act']))
        wic_pred = self.sigmoid(logits).tolist()
        wsd_pred = list(zip(result1['pred'].tolist(),result2['pred'].tolist()))

        return {'wsd_pred': wsd_pred, 'wic_pred': wic_pred}

    @torch.no_grad()
    def do_sentence_pairs_wsd(
        self,
        sentence_pairs: Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]],
        )-> Tuple[torch.Tensor,torch.Tensor]:

        dict1 = self.bert_wsd.eval().basic_step(sentence_pairs[0])
        dict2 = self.bert_wsd.eval().basic_step(sentence_pairs[1])

        return dict1, dict2
        
    def unpack_sentence_pairs(
        self,
        batch: Dict[str,List],
        ) -> Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]:

        batch1 = {
                'tokens': batch['tokens1'], 
                'key_indices': batch['key_indices1'],
                'synsets': batch['synsets']
                }
        batch2 = {
                'tokens': batch['tokens2'], 
                'key_indices': batch['key_indices2'],
                'synsets': batch['synsets']
                }

        return batch1, batch2
    
    def basic_step(
        self,
        batch: Dict[str,List],
        ) -> Dict[str,torch.Tensor]:
        """
        Evaluates performance on ground truth in terms of both loss (returned)
        and metrics are update
        """
        result1, result2 = self.do_sentence_pairs_wsd(self.unpack_sentence_pairs(batch))
        logits = self((result1['act'], result2['act']))
        pred = self.sigmoid(logits)
        gold = torch.tensor(batch['labels']).to(self.device)
        loss = self.loss(pred, gold.to(torch.float32))
        result = {'loss': loss, 'pred': pred, 'gold': gold }
        return result, result1, result2

    def training_step(
        self,
        batch: Dict[str,List],
        batch_idx: int
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        Computes loss to be used for .backward()
        """
        result = self.basic_step(batch)
        return result

    def write_metrics_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        metrics_wic: MetricCollection,
        metrics_wsd: MetricCollection,
        ):
        """
        Write metrics at end on multi GPUs
        """
        result, result1, result2 = batch_parts

        result_wsd = {
            'pred': torch.cat([result1['pred'], result2['pred']],dim=0),
            'gold': torch.cat([result1['gold'], result2['gold']],dim=0)
        }

        output = metrics_wsd(result_wsd['pred'],result_wsd['gold'])
        self.log_dict(output, on_step=True, on_epoch=False, prog_bar=True)

        self.log('loss_wic',result['loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('loss_wsd',(result1['loss']+result2['loss'])/2, on_step=True, on_epoch=False, prog_bar=True)

        output = metrics_wic(result['pred'],result['gold'])
        self.log_dict(output, on_step=True, on_epoch=False, prog_bar=True)

    def training_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward() on multi GPUs
        """
        self.write_metrics_end(batch_parts,self.train_metrics,self.bert_wsd.train_metrics)

        result, result1, result2 = batch_parts

        return result['loss']+result1['loss']+result2['loss']

    @torch.no_grad()
    def validation_step(
        self,
        batch: Dict[str,List],
        batch_idx: int
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        Evaluates on batch of validation samples
        """
        result = self.basic_step(batch)
        return result

    def validation_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward()
        """
        self.write_metrics_end(batch_parts,self.val_metrics,self.bert_wsd.val_metrics)

    @torch.no_grad()
    def test_step(
        self,
        batch: Dict[str,List],
        batch_idx: int
        ) -> Dict[str,torch.Tensor]:
        """
        [Required by lightning]
        Evaluates on batch of test samples
        """
        result = self.basic_step(batch)
        return result

    def test_step_end(
        self,
        batch_parts: List[Dict[str,torch.Tensor]],
        ):
        """
        [Required by lightning]
        Computes loss to be used for .backward()
        """
        self.write_metrics_end(batch_parts,self.test_metrics,self.bert_wsd.test_metrics)

    def configure_optimizers(self):
        """
        [Required by lightning]
        Initializes the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def basic_on_epoch_end(
        self,
        metrics: MetricCollection,
        ):
        """
        Log reduction of metrics
        """
        output = metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        """
        [lightning]
        Logging and EM reset (validation)
        """
        self.basic_on_epoch_end(self.train_metrics)

    def on_validation_epoch_end(self):
        """
        [lightning]
        Logging and EM reset (validation)
        """
        self.basic_on_epoch_end(self.val_metrics)

    def on_test_epoch_end(self):
        """
        [lightning]
        Logging and EM reset (test)
        """
        self.basic_on_epoch_end(self.test_metrics)


class HParams():
    hidden_size = 512
    dropout = 0.3
    vocab_size = len(SYNSETS)
    # initialize a fake graph which is loaded 
    coalesce_adjacency_matrix = get_adjacency_matrix()
    is_split_into_words = False
    train_adjacency = False
    lr = 5e-4
    wsd_checkpoint = None
    wsd_name = 'bert-large-uncased'
    wsd_freeze = True
    