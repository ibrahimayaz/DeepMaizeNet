import argparse
from turtle import color
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp

# Machine Learning
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K

# Helper libraries
import os
import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base", help="Hedef dizin", type=str, default="")
parser.add_argument("-w1", "--width", help="Hedef imgenin genisligi", type=int, default=256)
parser.add_argument("-h1", "--height", help="Hedef imgenin yukseligi", type=int, default=256)
parser.add_argument("-c", "--channel", help="Hedef imgenin kanal sayısı", type=int, default=1)
parser.add_argument("-e", "--epoch", help="Epoch sayısı", type=int, default=100)
parser.add_argument("-bs", "--batchsize", help="Yığın büyüklüğü", type=int, default=32)
parser.add_argument("-lr", "--learningrate", help="Öğrenme oranı", type=float, default=1e-4)
parser.add_argument("-pt", "--patience", help="Gelişmeyen Epoch Sayısı", type=int, default=20)
parser.add_argument("-kf", "--kfold", help="K-FOLD sayısı", type=int, default=5)
parser.add_argument("-s", "--seed", help="SEED sayısı", type=int, default=13)
args = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus, 'GPU')
strategy=tf.distribute.MirroredStrategy()

#--- HİPER PARAMETRELERİN TANIMLANMASI ---#
# Patience (Early Stopping)
PATIENCE = args.patience
# Target image shape
SHAPE = (args.width, args.height, args.channel)
# Training batch size
BATCH_SIZE = args.batchsize
# K-Fold splits
N_SPLITS = args.kfold
# Training epochs
EPOCHS = args.epoch
# Used for reproduction
SEED = args.seed
# Learning Rate
LEARNING_RATE = args.learningrate
# Labels
TARGET_NAMES = ["d", "h"]

# BASE Location Directory
BASE = args.base
# DATA Location Directory
DIR = BASE + "/dataset/all_data/"
# K-FOLD Model Path
MODEL_SAVE_PATH = BASE + "/models/"
# Tensorboard Records Path
LOGS_PATH = BASE + "/logs/"
# Confusion Matrix Path
CM_PATH = BASE + "/cm/"

#--- CSV Dosyasının Okunması ---#

data=pd.DataFrame()
data = pd.read_csv('data.csv')
color_mode="rgb"
if args.channel==1:
    color_mode="grayscale"

Y = data[['label']]
n = Y.size


class SGDRScheduler(Callback):
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / \
            (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * \
            (self.max_lr - self.min_lr) * \
            (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

#--- Performans Ölçümleri ---#

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))

def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())


#--- Modelin oluşturulması ---#
def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3),
               strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(
            1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y

def create_model():
    init = Input(SHAPE)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(init)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = residual_block(x, 32)
    x= MaxPooling2D((2, 2))(x)
    x1 = Dropout(0.45)(x)
    ginp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x1)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x2 = Dropout(0.45)(x)
    ginp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x2)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x2)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x3 = Dropout(0.45)(x)
    ginp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x3)

    hypercolumn = Concatenate()([ginp1, ginp2, ginp3])
    gap = GlobalAveragePooling2D()(hypercolumn)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(gap)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)

    x = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    y = Dense(2, activation='softmax')(x)

    model = Model(init, y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=[specificity, precision, recall, f1,  'acc'])
    return model


def plot_confusion_matrix(cm, target_names, idx, title='Hata Matrisi', cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Class')
    plt.xlabel('Predict Class \naccuracy={:0.4f}'.format(accuracy))
    plt.savefig(CM_PATH+"/confusion_matrix_"+str(idx)+".png", dpi=150)
    plt.show()




def calculate_tpr_fpr(_cm):
    # Calculates the confusion matrix and recover each element
    cm = _cm
    TP = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[1, 1]
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

#--- Cross Validation ve Early Stopping ---#
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)

#--- Eğitim Aşaması ---#

fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
fold_var = 1
for train_index, val_index in skf.split(np.zeros(n), Y):
    print("** FOLD {} **".format(fold_var))
    training_data = data.iloc[train_index]
    validation_data = data.iloc[val_index]
    td_generator = ImageDataGenerator(rescale=1./255, zoom_range=0.2, brightness_range=[0.2,1.0], rotation_range=90, fill_mode='nearest')
    train_data_generator = td_generator.flow_from_dataframe(training_data, directory=DIR, x_col='filename', y_col='label', class_mode="categorical", color_mode=color_mode, shuffle=True, batch_size=BATCH_SIZE, target_size=(SHAPE[0], SHAPE[1]))
    vd_generator = ImageDataGenerator(rescale=1./255)
    valid_data_generator = vd_generator.flow_from_dataframe(validation_data, directory=DIR, x_col='filename', y_col='label', class_mode="categorical", color_mode=color_mode, shuffle=False, batch_size=BATCH_SIZE, target_size=(SHAPE[0], SHAPE[1]))
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_SAVE_PATH, "model_fold_"+str(fold_var)+".h5"), monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)
    schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-3, steps_per_epoch=np.ceil(EPOCHS/BATCH_SIZE), lr_decay=0.8, cycle_length=10, mult_factor=1.)
    tensorboard = TensorBoard(log_dir=os.path.join(LOGS_PATH, 'log_'+str(fold_var)))
    callbacks = [model_checkpoint, tensorboard, schedule]
    with strategy.scope():
        model=create_model()
    history = model.fit(train_data_generator,
                        steps_per_epoch=len(train_index)//BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        validation_data=valid_data_generator,
                        validation_steps=len(val_index)//BATCH_SIZE, 
                        #use_multiprocessing=True,
                        callbacks=callbacks)
    #y_test
    true_y = valid_data_generator.classes
    test_steps_per_epoch = np.math.ceil(valid_data_generator.samples / valid_data_generator.batch_size)
    Y_pred = model.predict(valid_data_generator, verbose=0)
    y_preds = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    cnf_matrix = confusion_matrix(true_y, y_preds)
    print(cnf_matrix)
    plot_confusion_matrix(cm=cnf_matrix, normalize=False, target_names=TARGET_NAMES, idx=fold_var, title="Confusion Matrix")

    print('Classification Report')
    print(classification_report(true_y, y_preds, target_names=TARGET_NAMES))

    fpr, tpr, t = roc_curve(true_y, Y_pred[:,1])
    np.save(BASE+"/roc/fpr_"+str(fold_var)+".npy", fpr)
    np.save(BASE+"/roc/tpr_"+str(fold_var)+".npy", tpr)
    np.save(BASE+"/roc/treshold_"+str(fold_var)+".npy", t)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (AUC = %0.2f)' % (fold_var, roc_auc))


    fold_var += 1
#--- Eğitim Geçmişinin Kaydedilmesi ---#

plt.plot([0,1],[0,1],linestyle = '--',lw = 1,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=1.5, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('ROC',dpi=300)
