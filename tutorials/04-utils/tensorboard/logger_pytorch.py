from io import BytesIO         # Python 3.x
import numpy as np
from PIL import Image 
import torch 
import torch.utils.tensorboard as tb
import torchvision

class Logger(object):
    def __init__(self, log_dir):
        """ Create a summary writer object logging to log_dir."""
        self.writer = tb.SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag,value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step,dataformats='HWC'):
        """Log image , Input image will be a numpy ndarray 
        dataformats : can be changed as per requirement to HWC,CHW..
        where H-Height W-Width and C-Channels of image"""
        self.writer.add_image(tag,image,step,dataformats)
        self.writer.flush()	
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag,values, step)
        self.writer.flush()
   
    def text_summary(self,tag,value,step):
        """Log text with tag to it"""
        self.writer.add_text(tag,value,step)
        self.writer.flush()

    def embedding_summary(self,embedding_matrix, metadata=None, label_img=None, 
          global_step=None, tag='default', metadata_header=None):
        """Log embedding matrix to tensorboard."""
        self.writer.add_embedding(embedding_matrix, metadata, label_img,global_step, tag,
                metadata_header)
        self.writer.flush()

    def plot_pr_summary(self,tag, labels, predictions, global_step=None,
            num_thresholds=127, weights=None, walltime=None):
        """Plot Precision/Recall curves with labels being actual labels 
        and predictions being how accurarte(in tems of %)"""
        self.writer.add_pr_curve(tag, labels, predictions, global_step, num_thresholds, weights, walltime)
        self.writer.flush()
   
    def __del__(self):
        """close the writer"""
        self.writer.close()
