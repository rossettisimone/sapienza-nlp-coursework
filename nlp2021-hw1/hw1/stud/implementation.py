# general
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import re
from typing import *
import torch
from model import Model
from torch import nn
from random import randint
# nltk 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')
stop_words = set(nltk.corpus.stopwords.words('english'))

# here we define NE labels for some further experiments
NER_LABELS = ['ORGANIZATION', 'PERSON', 'GSP', 'GPE', \
        'LOCATION', 'FACILITY'] 

# Here we define POS+NE tags wich we will use for some further experiments
POS_NE_TAGS = {
            'NOUN': 0,
            'VERB': 1,
            'ADJ': 2,
            'ADV': 3,
            'GPE': 4,
            'ORGANIZATION': 5,
            'PERSON': 6,
            'LOCATION': 7,
            'FACILITY': 8,
            'GSP': 9,
            '<pad>': 10,
            '<drop>': 11,
            '<sep>': 12
            }

# these tokens are special positions in our embedding tensor
RESERVED_TOKENS = {'<pad>': 0, # padding
                    '<unk>': 1, # unknown
                    '<sep>': 2, # sentences separation
                    '<drop>': 3, # dropped word
                    '<numb>': 4, # number
                    '<punct>': 5} # punctuation


def build_model(device: str) -> Model:

    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()
    model = StudentModel().to(device)
    model.device = device
    return model


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model, nn.Module):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self) -> None:
        super(StudentModel, self).__init__()
        self.n_hidden = 256
        self.vectors_store_size = 400005
        self.embed_dim = 300
        # these tokens are special positions in our embedding tensor
        self.embedding = torch.nn.Embedding(self.vectors_store_size, self.embed_dim)
        self.word_index = defaultdict(lambda: RESERVED_TOKENS['<unk>'], dict())
        # sequence encoder of size 301: 300 word embedding size + 1 one hot 
        # encoding of the position of query lemma
        self.rnn = torch.nn.GRU(input_size=self.embed_dim+1, hidden_size=self.n_hidden, \
             num_layers=2, batch_first=True, dropout=0.2, bidirectional=False)
        # classification head
        self.norm1 = torch.nn.BatchNorm1d(self.n_hidden*2)
        self.fc1 = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.drop1 = nn.Dropout(p=0.2)
        self.norm2 = torch.nn.BatchNorm1d(self.n_hidden)
        self.fc2 = torch.nn.Linear(self.n_hidden, 1)
        self.drop2 = nn.Dropout(p=0.2)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.noise_mean = 0.0
        self.noise_std = 1.0 # see training code
        self.load_pretrained()

    def forward(
        self,
        X: torch.Tensor,
        X_indices: torch.Tensor,
        X_length: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = X.shape
        # let's find the sep token and divide the sentences
        sep_token = torch.where(X[0]==RESERVED_TOKENS['<sep>'])[0]

        # here I implemented dropwords
        if y is not None: 
            bool_mask = torch.empty(batch_size, seq_len).uniform_(0, 1).to(self.device) > 0.5
            mask = torch.ones(batch_size, seq_len).type(torch.LongTensor).to(self.device)
            X = torch.where(bool_mask, X, RESERVED_TOKENS['<drop>']*mask)
        
        # here we can separate the sentences
        X1 = X[:,:sep_token]
        X2 = X[:,sep_token+1:]

        # embedding words from indices
        embedding_out1 = self.embedding(X1)
        embedding_out2 = self.embedding(X2)

        # here I added noise to the input to improve generalization
        if y is not None:
            embedding_out1 +=  torch.normal(self.noise_mean, \
                1.5*self.noise_std, size=embedding_out1.shape).to(self.device)
            embedding_out2 +=  torch.normal(self.noise_mean, \
                1.5*self.noise_std, size=embedding_out2.shape).to(self.device)

        batch_size, seq_len1, _ = embedding_out1.shape
        _, seq_len2, _ = embedding_out2.shape

        # here I encode the indices of the query lemma
        target1 = torch.nn.functional.one_hot(X_indices[...,0], num_classes=seq_len1)\
            .to(torch.float32).unsqueeze(-1)
        target2 = torch.nn.functional.one_hot(X_indices[...,1], num_classes=seq_len2)\
            .to(torch.float32).unsqueeze(-1)

        # encode the sequence
        recurrent_out1, _ = self.rnn(torch.cat([embedding_out1, target1],dim=-1))
        recurrent_out2, _ = self.rnn(torch.cat([embedding_out2, target2],dim=-1))

        # here we utilize the sequences length to retrieve the last token
        # output for each sequence
        batch_size, seq_len1, hidden_size = recurrent_out1.shape
        _, seq_len2, _ = recurrent_out2.shape
        # we flatten the recurrent output
        # now I have a long sequence of batch x seq_len vectors
        flattened_out1 = recurrent_out1.reshape(batch_size * seq_len1, hidden_size)
        flattened_out2 = recurrent_out2.reshape(batch_size * seq_len2, hidden_size)
        # tensor of the start offsets of each element in the batch
        sequences_offsets1 = torch.arange(batch_size, device=self.device) * seq_len1
        sequences_offsets2 = torch.arange(batch_size, device=self.device) * seq_len2
        # and we use a simple trick to compute a tensor of the indices
        # of the last token in each batch element
        vect1 = sequences_offsets1 + X_length[...,0]-1
        vect2 = sequences_offsets2 + X_length[...,1]-1
        
        # we retreive the output of the last token
        out1 = flattened_out1[vect1]
        out2 = flattened_out2[vect2]

        # we concatenate the encoded sequences
        # and send them to the classifier
        out = torch.cat([out1, out2],dim=-1)
        out = self.norm1(out)
        out = self.drop1(out)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.norm2(out)
        out = self.drop2(out)
        logits = self.fc2(out)
        pred = torch.sigmoid(logits)

        result = {}
        result['pred'] = pred
        result['logits'] = logits

        # compute loss
        if y is not None:
            loss = self.loss(logits, y.to(torch.float32))
            result['loss'] = loss

        return result

    def load_pretrained(self) -> None:
        checkpoint = torch.load('./model/best-gru2-parameters-.697.pt', map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.word_index = defaultdict(lambda: RESERVED_TOKENS['<unk>'], checkpoint['word_index']) 
        self.eval()

    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!        
        # pass
        outputs = []
        for sample in sentence_pairs:
            X, indices, lengths = preprocess_sample(self.word_index, sample)
            output = self(X.unsqueeze(0), indices.unsqueeze(0), lengths.unsqueeze(0))['pred'].squeeze(0)
            prediction = 'True' if output.round() else 'False'
            outputs.append(prediction)
        return outputs

# PREPROCESSING help functions

def sentence2indices(words_tags: List[Tuple[str,str]], dictionary: defaultdict) -> torch.Tensor:
    ''' Convert words to words indices '''
    return torch.tensor([dictionary[word] for word, tag in words_tags], dtype=torch.long)
    
def preprocess_sample(word_index: defaultdict, sample: Dict) \
    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    '''
    Preprocess a sentence and return a tokenized sequence of word and tags indices
    '''
    sentence1 = sample['sentence1']
    sentence2 = sample['sentence2']
    lemma = sample['lemma']
    start1 = int(sample['start1'])
    start2 = int(sample['start2'])
    end1 = int(sample['end1'])
    end2 = int(sample['end2'])
    pos = sample['pos']
    
    words_tags1, index1 = preprocess_sentence(sentence1, lemma, pos, start1, end1)
    words_tags2, index2 = preprocess_sentence(sentence2, lemma, pos, start2, end2)
    
    words1 = sentence2indices(words_tags1, word_index)
    words2 = sentence2indices(words_tags2, word_index)
    indices = torch.tensor([index1, index2])
    
    tokens = torch.cat([words1, torch.tensor([RESERVED_TOKENS['<sep>']]), words2],dim=-1)
    
    lenghts = torch.tensor([len(words1), len(words2)])
    return tokens, indices, lenghts

def get_words(text: str) -> List[Tuple[str, str]]:
    '''
    Perform the canonical parsing of a sentence plus some extra rules:
    1. TOKENIZATION (- STOPWORDS)
    2. POS TAGGING (UNIVERSAL POS)
    3. LEMMATIZATION
    Returns:
        - words: processed words
        - tags: relative POS tags
    '''
    parsed = []
    text = str(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0–9^,!.\/’+-=]", " ", text)
    text = re.sub(r"(\d+)\,(\d+)", r"\g<1>\g<2>", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\g<1>\g<2>", text)
    text = re.sub(r"(\d+)\–(\d+)", r"\g<1> \g<2>", text)
    text = re.sub(r"what’s", " what is ", text)
    text = re.sub(r"What’s", " What is ", text)
    text = re.sub(r"\’s", " ", text)
    text = re.sub(r"\’ve", " have ", text)
    text = re.sub(r"can’t", " can not ", text)
    text = re.sub(r"Can’t", " Can not ", text)
    text = re.sub(r"won't", " will not ", text)
    text = re.sub(r"Won't", " will not ", text)
    text = re.sub(r"n’t", " not ", text)
    text = re.sub(r"i’m", " i am ", text)
    text = re.sub(r"I’m", " I am ", text)
    text = re.sub(r"\’re", " are ", text)
    text = re.sub(r"\’d", " would ", text)
    text = re.sub(r"\’ll", " will ", text)
    text = re.sub(r"\;", " ", text)
    text = re.sub(r"\,", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"\!", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\°", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\—", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"\’", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r"\:", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" E g ", " eg ", text)
    text = re.sub(r" U S ", " american ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", text)
    text = re.sub(r"(\d+)([A-Za-z]+)", r"\g<1> \g<2>", text)
    text = re.sub(r"([A-Za-z]+)(\d+)", r"\g<1> \g<2>", text)
    for tokenized in sent_tokenize(text, language='english'):
        # Word tokenizers is used to find the words and punctuation in a string
        words_list = nltk.word_tokenize(tokenized)
        # Using a POS Tagger
        words_pos = nltk.pos_tag(words_list)
        words_pos = [(w.lower(),t) for (w,t) in words_pos if (not w.lower() in stop_words)]
        words_pos = [(w,nltk.tag.mapping.map_tag('en-ptb', 'universal', t)) for (w,t) in words_pos ]
        words_pos = [(w,t) for (w,t) in words_pos if t in POS_NE_TAGS and len(w) > 1]
        parsed.extend(words_pos)
    return parsed 
    

def preprocess_sentence(sentence: str, lemma: str, pos: str, start: int, end:int) \
    -> Tuple[List[Tuple[str,str]],int]:
    '''
    Map a WiC format sample to a the get_words function, 
    Returns:
        - (words, tags): list of processed words in the sentence, POS tagging of the sentence
        - index: position of the query lemma
    '''
    words_tags_list = get_words(sentence[:start]) + [(lemma, pos)] + get_words(sentence[end:]) # process pre word sentence
    words = []
    tags = []
    for word, tag in words_tags_list:
        words.append(word)
        tags.append(tag)
    # index = words.index(lemma)
    if lemma in words:
        index = words.index(lemma)
        words_tags_list[index] = (lemma, pos)
    else:
        index = randint(0,len(words_tags_list))
        words_tags_list = words_tags_list[:index] + \
            [(lemma, pos)] + words_tags_list[index:] 
    return words_tags_list, index

