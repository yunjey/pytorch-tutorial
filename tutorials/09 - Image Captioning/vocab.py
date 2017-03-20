import nltk
import pickle
import os
from configuration import Config
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Builds a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))
    
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>') 
    vocab.add_word('<start>') 
    vocab.add_word('<end>') 
    vocab.add_word('<unk>') 
    
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    config = Config()
    vocab = build_vocab(json=os.path.join(config.caption_path, 'captions_train2014.json'),
                        threshold=config.word_count_threshold)
    vocab_path = os.path.join(config.vocab_path, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved the vocabulary wrapper to ", vocab_path)
    
if __name__ == '__main__':
    main()