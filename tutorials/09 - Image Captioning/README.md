## Usage 


#### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone https://github.com/yunjey/pytorch-tutorial.git
$ cd pytorch-tutorial/tutorials/09\ -\ Image\ Captioning
```

#### 2. Download the dataset

```bash
$ pip install -r requirements.txt
$ chmod +x download.sh
$ ./download.sh
```

#### 3. Preprocessing

```bash
$ python build_vocab.py   
$ python resize.py
```

#### 4. Train the model

```bash
$ python train.py    
```

#### 5. Generate captions


```bash
$ python sample.py --image='path_for_image'
```

<br>

## Pretrained model 

If you do not want to train the model yourself, you can use a pretrained model. I have provided the pretrained model as a zip file. You can download the file [here](https://www.dropbox.com/s/b7gyo15as6m6s7x/train_model.zip?dl=0) and extract it to `./models/` directory.
