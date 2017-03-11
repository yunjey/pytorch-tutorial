# Create a vocabulary wrapper
import nltk
import pickle
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
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if i % 1000 == 0:
            print("[%d/%d] tokenized the captions." %(i, len(ids)))
        
    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>') 
    vocab.add_word('<start>') 
    vocab.add_word('<end>') 
    vocab.add_word('<unk>') 
    
    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    vocab = create_vocab(json='./data/annotations/captions_train2014.json',
                         threshold=4)
    with open('./data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './data/vocab.pkl')
    
if __name__ == '__main__':
    main()