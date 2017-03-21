import torchvision.transforms as T 


class Config(object):
    """Wrapper class for hyper-parameters."""
    def __init__(self):
        """Set the default hyper-parameters."""
        # Preprocessing
        self.image_size = 256
        self.crop_size = 224
        self.word_count_threshold = 4
        self.num_threads = 2
        
        # Image preprocessing in training phase
        self.train_transform = T.Compose([
            T.Scale(self.image_size),    
            T.RandomCrop(self.crop_size),
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # Image preprocessing in test phase
        self.test_transform = T.Compose([
            T.Scale(self.crop_size),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # Training 
        self.num_epochs = 5
        self.batch_size = 64
        self.learning_rate = 0.001
        self.log_step = 10
        self.save_step = 1000
        
        # Model
        self.embed_size = 256
        self.hidden_size = 512
        self.num_layers = 2
        
        # Path 
        self.image_path = './data/'
        self.caption_path = './data/annotations/'
        self.vocab_path = './data/'
        self.model_path = './model/'
        self.trained_encoder = 'encoder-4-6000.pkl'
        self.trained_decoder = 'decoder-4-6000.pkl'