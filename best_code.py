from tensorflow.keras.models import Sequential     #선형회기 모델을 불러온다
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,MaxPooling2D,Flatten,Conv2D,Activation,BatchNormalization,AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator        #imagedatagenerator을 불러온다
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import VGG16     #특징 추출을 하기 위해 필요한 코드
from keras import optimizers
from tensorflow.keras.optimizers import Adadelta,Adagrad,RMSprop,Nadam
from time import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
#모델 개선 버전-빨간색을 바꾼 모델==>저번 모델보다는 빠르게 학습률을 올릴 수 있음(안정적으로 70%룰 넘을 수 있음,loss는 10%대로 내려감)
#
os.environ['TF_MIN_LOG_LEVEL']='2'
#초기 가중치 설정=>1.평균은 0 2.분산은 모든 레이어에서 동일하게 유지

def cnn_classifer():
    classifer=Sequential()      #모델 생성
    #해야 할 일: 모델을 chalkpoint를 정하여 저장하기, 가능하다면 병목특징 만들기, 학습률 높이는 알고리즘 찾아서 적용하기

    #1
    classifer.add(Conv2D(filters=20, kernel_size=(3, 3), activation = 'relu',input_shape = (64, 64, 1),padding='same',strides=(2,2),kernel_initializer='he_normal'))
    classifer.add(BatchNormalization())

    classifer.add(Activation('relu'))
    classifer.add(AveragePooling2D(pool_size=(2,2)))      #globalmaxpooling으로 고침(원래는 maxpooling)
    #classifer.add(BatchNormalization())

    classifer.add(Conv2D(filters=20,kernel_size=(3,3),activation='relu',padding='same',strides=(2,2),kernel_initializer='he_normal'))
    #classifer.add(BatchNormalization())

    classifer.add(MaxPooling2D(pool_size=(2,2)))
    #classifer.add(BatchNormalization())

    classifer.add(Dropout(0.25))

    classifer.add(Flatten())        #2차원으로 변환(항상 맨 아래에 있어야 함)


    #세번째. Flatten층 추가하기=>1D로 변환
    #네번째, Dense층 추가하기
    classifer.add(Dense(units=32,activation='relu',kernel_initializer='he_normal'))
    classifer.add(Dropout(0.5))
    classifer.add(Dense(units=1,activation='hard_sigmoid',kernel_initializer='he_normal'))        #시그모이드 함수는 바꾸지 말것
    #이진분류 필요
    classifer.summary()

    optimizer=keras.optimizers.Adadelta(rho=0.95,lr=1.0,decay=0.0)            #학습의 단계가 진행될 수록 계수의 값이 작아지는 문제를 해결
                                            #모든 경사의 제곱합을 평균 감쇄하여 재곱식으로 계산
    sgd=optimizers.SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)       #학습률을 처음에는 크게하였다 점점 줄임
    #다섯번째, 모델 컴파일하기
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # optimizer에 adagrad를 넣는다면 적응형 함수
    #classifer.compile(loss="mse",metrics=["mae"])
    classifer.compile(optimizer=tf.keras.optimizers.Adadelta(rho=0.95,lr=1.0,decay=0.0),
                      loss='mse',metrics=['accuracy'])      #loss를 binary_crossentropy로 쓰기도 함/무조건 mse를 사용하여야 손실이 줄어듬

    #classifer.compile(optimizer=SGD(lr=0.1,momentum=0.9,nesterov=True,metrics=['accuracy']
                     # loss='mse',metrics=['accuracy'])
    #케라스 모델 별 optimizer정리
    #sgd:확률적 경사 하강법/momentum학습률을 고정하고 매개변수를 조정
    #adagrad:학습률을 조정하여 학습
    #rmsprop,adadelta:adagrad를 보완
    #adam이 기본, spare~은 인코딩 필요없이 바로 가능, 측정함수는 모델의 성능 평가=>손실함수 사용가능

    return classifer

#추가할 코드:1.이미지 변형,2.과적합 방지 코드
model=cnn_classifer()

#데이터 시험하기

train_datagen= ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

batch_size=32
test_datagen=ImageDataGenerator(rescale=1./255,)

train_set=train_datagen.flow_from_directory('data_pink',target_size=(64,64),color_mode='grayscale',shuffle=False,
                                            batch_size=batch_size,class_mode='binary')
#class_mode='binary'는 1차원의 이진 정수라벨이 반환(절대 바꾸지 말것!!!)
#디렉토리의 위치를 전달받아 정규화된 데이터의 배치를 생성
test_set=test_datagen.flow_from_directory('data_pink_test',target_size=(64,64),color_mode='grayscale',
                                          batch_size=batch_size,class_mode='binary')
#이미지의 데이터 프레임(디렉토리)    (테스트셋 만들기)
# #이미지의 크기를 재조정
# #이미지 배치의 크기
# #일차원 배열
early_stopping=keras.callbacks.EarlyStopping(monitor='acc',patience=1,mode='max',verbose=0)       #손실이 가장 적도록 설정
writer=TensorBoard(log_dir='\Desktop\pythonProject\logs'.format(time()))

#콜백함수로 학습률 조정
reduceLR=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5)
hist=model.fit_generator(train_set,steps_per_epoch=batch_size,epochs=50,validation_data=test_set,
                        validation_steps=batch_size,callbacks=[reduceLR,early_stopping])          #earlystopping을 추가
#인터넷을 찾아보니 validation_step과 steps_per_epoch는 batch_size와 같아야 된다고 함

#erarly_Stopping함수를 추가해야함
#그래프를 기르기 위한 설정
acc=hist.history['acc']
val_acc=hist.history['val_acc']
loss=hist.history['loss']
val_loss=hist.history['val_loss']
epoches=range(len(acc))
#그래프를 그리는 코드
plt.plot(epoches,acc,'bo',label='training acc')
plt.plot(epoches,val_acc,'b',label='validation acc')
plt.title('training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epoches, loss, 'bo', label='Training loss')
plt.plot(epoches, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
print(hist.history)
#위 에러 때문에
model.save('model_again.h5')

output1=model.predict_generator(test_set,steps=1)           #이 코드는 절대 바꾸지 말것!!
result1=test_set.class_indices
print(result1)
print(output1)
#tensorboard를 통해 그래프 그리기(tensorboard dev upload --logdir \)
#데이터셋 수는 약 합쳐서 900장