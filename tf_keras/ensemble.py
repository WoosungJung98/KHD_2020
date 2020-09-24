import os, sys
import argparse
import time
import random
import cv2
import numpy as np
import keras

from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.layers import BatchNormalization, ReLU
from keras.initializers import TruncatedNormal
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K

# from imblearn.over_sampling import SMOTE

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM


IMSIZE = 331, 331
VAL_RATIO = 0.1
RANDOM_SEED = 1234



def bind_model(effb0_224, effb0_224_ratio,
               effb1_224, effb1_224_ratio,
               effb5_224, effb5_224_ratio
               ):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        effb0_224.save_weights(os.path.join(dir_name, 'effb0_224'))
        effb1_224.save_weights(os.path.join(dir_name, 'effb1_224'))
        effb5_224.save_weights(os.path.join(dir_name, 'effb5_224'))
        print('model saved!')

    def load(dir_name):
        effb0_224.load_weights(os.path.join(dir_name, 'effb0_224'))
        effb1_224.load_weights(os.path.join(dir_name, 'effb1_224'))
        effb5_224.load_weights(os.path.join(dir_name, 'effb5_224'))
        print('model loaded!')

    def infer(data):            # test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = ImagePreprocessing(data, (224, 224))
        X = np.array(X)
        
        effb0_224_pred = effb0_224.predict(X) * effb0_224_ratio
        effb1_224_pred = effb1_224.predict(X) * effb1_224_ratio
        effb5_224_pred = effb5_224.predict(X) * effb5_224_ratio

        pred = effb0_224_pred + effb1_224_pred + effb5_224_pred

        pred = np.argmax(pred, axis=1)
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


def Class2Label(cls):
    lb = [0] * 4
    lb[int(cls)] = 1
    return lb

def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_]
        r_img = img_whole[:, :w_]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img);      lb.append(Class2Label(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img);      lb.append(Class2Label(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb


def ImagePreprocessing(img, img_size=IMSIZE):
    # 자유롭게 작성
    h, w = img_size
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        origin_shape = im.shape

        # Crop
        origin_h, origin_w = origin_shape[0], origin_shape[1]
        start_h = (origin_h - origin_w) // 2

        im = im[
            start_h:start_h + origin_w,
            :
        ]
        origin_shape = im.shape

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(im)

        # Contours
        _, threshold = cv2.threshold(clahe_img, 127, 255, 0)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        base = np.zeros_like(im)
        contour_img = cv2.drawContours(base, contours, -1, 255, 1)

        # Merge 3 Images to make 3 Channel 
        im = np.concatenate([
            im.reshape((origin_shape[0], origin_shape[1], 1)),
            clahe_img.reshape((origin_shape[0], origin_shape[1], 1)),
            contour_img.reshape((origin_shape[0], origin_shape[1], 1))
        ], axis=-1)
        # Resize
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)

        if len(tmp.shape) == 2:
            tmp = np.expand_dims(tmp, axis=2)
        tmp = tmp / 255.
        img[i] = tmp

    print(len(img), 'images processed!')
    return img


def imagenet(imagenet_class, in_shape, num_classes):
    base_model = imagenet_class(include_top=False,
                                weights=None, #"imagenet",
                                input_shape=in_shape)

    # base_model.trainable = False
    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    initializer = TruncatedNormal(stddev=0.01)
    dropout_rate = 0.5
    dense_layers = [64, 32]
    for node in dense_layers:
        x = Dense(node, activation="relu", kernel_initializer=initializer)(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=x)


def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=10)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-4)  # learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    args.add_argument('--name', type=str, default="")     # 구분을 위한 이름 설정

    args.add_argument('--effb0_224', type=str, default="KHD007/2020KHD_PNS/43")
    args.add_argument('--effb1_224', type=str, default="KHD007/2020KHD_PNS/45")
    args.add_argument('--effb5_224', type=str, default="KHD007/2020KHD_PNS/51")

    args.add_argument('--effb0_224_ratio', type=float, default=1.)
    args.add_argument('--effb1_224_ratio', type=float, default=1.)
    args.add_argument('--effb5_224_ratio', type=float, default=1.)

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config


if __name__ == '__main__':
    args = argparse.ArgumentParser()


    config = ParserArguments(args)
    nb_epoch, batch_size, num_classes, learning_rate, ifpause, ifmode = config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.pause, config.mode

    seed = 1234
    np.random.seed(seed)

    """ Model """
    h, w = IMSIZE
    in_shape_224 = (224, 224, 3)
    
    # model = SampleModelKeras(in_shape=in_shape, num_classes=num_classes)

    import efficientnet.keras as efn

    effb0_224 = imagenet(efn.EfficientNetB0, in_shape_224, num_classes)
    effb1_224 = imagenet(efn.EfficientNetB1, in_shape_224, num_classes)
    effb5_224 = imagenet(efn.EfficientNetB5, in_shape_224, num_classes)

    optimizer = optimizers.Adam(lr=learning_rate, decay=1e-5)    
    effb0_224.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    effb1_224.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    effb5_224.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    def effb0_224_load(dir_name):
        effb0_224.load_weights(os.path.join(dir_name, 'model'))
        print('effb0_224 loaded')
    
    def effb1_224_load(dir_name):
        effb1_224.load_weights(os.path.join(dir_name, 'model'))
        print('effb1_224 loaded')

    def effb5_224_load(dir_name):
        effb5_224.load_weights(os.path.join(dir_name, 'model'))
        print('effb5_224 loaded')

    # checkpoint 99로 고정
    nsml.load(checkpoint=99, load_fn=effb0_224_load, session=config.effb0_224)
    nsml.load(checkpoint=99, load_fn=effb1_224_load, session=config.effb1_224)
    nsml.load(checkpoint=99, load_fn=effb5_224_load, session=config.effb5_224)

    bind_model(effb0_224, config.effb0_224_ratio,
               effb1_224, config.effb1_224_ratio,
               effb5_224, config.effb5_224_ratio,
               )

    if ifpause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ### training mode일 때
        print('Training Start...')
        images, labels = DataLoad(os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images, (224, 224))
        ## data 섞기
        images = np.array(images)
        # images = np.expand_dims(images, axis=-1)
        labels = np.array(labels)
        dataset = [[X, Y] for X, Y in zip(images, labels)]
        random.shuffle(dataset)
        X = np.array([n[0] for n in dataset])
        Y = np.array([n[1] for n in dataset])

        # sm = SMOTE(random_state=124)

        # indices = np.expand_dims(np.arange(len(Y)), axis=-1)
        # indices, Y = sm.fit_resample(indices, Y)
        # X = X[np.squeeze(indices)]

        # print("Resampled", X.shape, Y.shape)

        """ Callback """
        monitor = 'categorical_accuracy'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = len(X) // batch_size
        print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))

        ## data를 trainin과 validation dataset으로 나누기
        tmp = int(len(Y) * VAL_RATIO)
        X_train, Y_train = X[tmp:], Y[tmp:]
        X_val, Y_val = X[:tmp], Y[:tmp]

        effb0_224_pred = effb0_224.predict(X) * config.effb0_224_ratio
        effb1_224_pred = effb1_224.predict(X) * config.effb1_224_ratio
        effb5_224_pred = effb5_224.predict(X) * config.effb5_224_ratio

        pred = effb0_224_pred + effb1_224_pred + effb5_224_pred

        pred = np.argmax(pred, axis=1)

        def GetF1score(y, y_pred, target):
            tp = 0
            fp = 0
            fn = 0
            for i, y_hat in enumerate(y_pred):
                if (y[i] == target) and (y_hat == target):
                    tp += 1
                if (y[i] == target) and (y_hat != target):
                    fn += 1
                if (y[i] != target) and (y_hat == target):
                    fp += 1

            try:
                f1s = tp / ( tp + (fp + fn)/2 )
            except ZeroDivisionError:
                f1s = 0
            
            return f1s

        def CategoricalF1Score(y, y_pred, num_classes):
            F1scores = []
            for t in range(num_classes):
                F1scores.append(GetF1score(y, y_pred, str(t)))
            return F1scores

        score = CategoricalF1Score(pred, Y_val, num_classes)

        nsml.report(summary=True, step=0, epoch_total=0, score=score)
        nsml.save(0)
