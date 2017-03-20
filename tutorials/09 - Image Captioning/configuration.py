class Config(object):
    """Wrapper class for hyper-parameters."""
    def __init__(self):
        """Set the default hyper-parameters."""
        # Preprocessing
        self.image_size = 256
        self.crop_size = 224
        self.word_count_threshold = 4
        self.num_threads = 2
        
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