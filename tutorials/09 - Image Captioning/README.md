## Usage 


#### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
$ git clone https://github.com/yunjey/pytorch-tutorial.git
$ cd pytorch-tutorial/tutorials/09 - Image Captioning
```

#### 2. Download the dataset

```bash
$ pip install -r requirements
$ chmod +x download.sh
$ ./donwload.sh
```

#### 3. Preprocessing

```bash
$ python vocab.py    
```

#### 4. Train the model

```bash
$ python train.py    
```

#### 5. Generate captions


```bash
$ python sample.py --image=sample_image.jpg     
```

<br>

## Pretrained model 

If you do not want to train the model yourself, you can use a pretrained model. I have provided the pretrained model as a zip file. You can download the file [here](https://www.dropbox.com/s/cngzozkk73imjdh/trained_model.zip?dl=0) and extract it to `model` directory.
