# Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
# Numpy
import numpy as np

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from util import base64_to_pil
import os

app = Flask(__name__)

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
    init = Input((256,256,3))
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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=10e-4))
    return model


labels=["Diploid", "Haploid"]
MODEL_PATH = 'orijinal/models/model_fold_5.h5'
model=create_model()
#Eğitilen modelin yüklenmesi
model = load_model(MODEL_PATH, compile = False)
model.make_predict_function()          
print('Model yüklendi.')

def model_predict(img, model):
    img = image.load_img(img,target_size=(256,256,3))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = img * 1./ 255

    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)

        img.save("./uploads/image.png")

        # yüklenen resmi tahmin etme
        preds = model_predict("./uploads/image.png", model)
        print(preds)
        # tahmin edilen sonucun değerini hesaplama
        pred_proba = "{:.3f}".format(np.amax(preds)) 
        # tahmin edilen değerin etiketini belirleme
        result=str("{:.2f}".format(np.float64(pred_proba)*100))+"%" + " " + labels[np.argmax(preds)] 
        os.remove("./uploads/image.png")
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)
