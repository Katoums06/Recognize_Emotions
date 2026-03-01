import torch

class Config:
    TRAIN_PATH = 'data\\train'
    TEST_PATH = 'data\\test'

    BATCH_SIZE = 128
    IMAGE_SIZE = 48

    CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_TO_IDX = {
                'angry' : 0,
                'disgust' : 1,
                'fear' : 2,
                'happy' : 3,
                'neutral' : 4,
                'sad' : 5,
                'surprise': 6
            }