import os
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader.
    
    This is just for tutorial. You can use the prebuilt torchvision.datasets.ImageFolder.
    """
    def __init__(self, root, transform=None):
        """Initializes image paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

    
def get_loader(image_path, image_size, batch_size, num_workers=2):
    """Builds and returns Dataloader."""
    
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = ImageFolder(image_path, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader