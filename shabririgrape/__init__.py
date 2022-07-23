import os


class DATASET:
    ROOT = 'diabetic_retinopathy_dataset/'
    METADATA = os.path.join(ROOT, 'metadata')
    IMGS = os.path.join(ROOT, 'data')
    EXT = '.jpeg'

    TRAIN_IMG = os.path.join(METADATA, 'train_img.csv')
    TRAIN_LABEL = os.path.join(METADATA, 'train_label.csv')
    TEST_IMG = os.path.join(METADATA, 'test_img.csv')
    TEST_LABEL = os.path.join(METADATA, 'test_label.csv')

    NUM_CLASSES = 5


class MODE:
    TRAIN = 'train'
    TEST = 'test'


class MODEL_TYPE:
    RESNET18 = 'resnet18'
    RESNET50 = 'resnet50'

    WITH_PRETRAINED = 'with'
    WITHOUT_PRETRAINED = 'without'
    FIXED_PRETRAINED = 'fixed'

    ADAM = 'adam'
    SGD = 'sgd'

    CROSS_ENTROPY = 'cross_entropy'


class DEVICE_TYPE:
    CPU = 'cpu'
    CUDA = 'cuda'
