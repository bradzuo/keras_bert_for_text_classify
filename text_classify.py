"""
author: long.zuo
time:2020.6.2
model:Bert
"""

# coding:utf-8
import numpy as np
from keras.optimizers import Adam
from keras_bert import load_vocabulary,Tokenizer,load_trained_model_from_checkpoint
from config import Config
import argparse
from keras.layers import Dense,Dropout,Lambda,Input
from keras.models import Model
from keras.callbacks import EarlyStopping,CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import KFold
from keras.utils import to_categorical

CONFIG = Config()
parse = argparse.ArgumentParser()
parse.add_argument('-d','--DATA_PATH',default=CONFIG.data_file,help='文本数据路径')
parse.add_argument('-l','--LABELS',default=CONFIG.labels,help='文本总的类别')
parse.add_argument('-t','--MAX_LEN',default=CONFIG.max_text_length,help='文本最大句长')
parse.add_argument('-b','--BATCH_SIZE',default=CONFIG.batchsize,help='batchsize')
parse.add_argument('-e','--EPOCH',default=CONFIG.epoch,help='epoch')
parse.add_argument('-c','--BERT_CONFIG',default=CONFIG.bert_config,help='bert配置文件')
parse.add_argument('-bm','--BERT_MODEL',default=CONFIG.bert_model,help='bert模型ckpt')
parse.add_argument('-v','--VOCAB',default=CONFIG.vocab,help='bert词汇表')
parse.add_argument('-m','--MODEL_PATH',default=CONFIG.bert_model,help='模型保存路径')
parse.add_argument('-lg','--LOG_PATH',default=CONFIG.bert_model,help='训练日志路径')
args = parse.parse_args()

# 加载bert分词器
token_dict = load_vocabulary(vocab_path=args.VOCAB)
tokenizer = Tokenizer(token_dict=token_dict)

def data_padding(data,padding=0):
    """
    数据padding
    :param data:
    :param padding:
    :return:
    """
    data_len = [len(d) for d in data]
    M_L = max(data_len)
    return np.array(
        [np.concatenate([d,(M_L-len(d))*[padding]]) if len(d) < M_L else d for d in data]
    )

class DataGenerator():

    def __init__(self,data,batchsize=args.BATCH_SIZE):
        self.data = data
        self.batchsize = batchsize
        self.step = len(self.data)//self.batchsize
        if len(data) % self.batchsize !=0:
            self.step += 1

    def __iter__(self,):
        """
        数据迭代生成器
        :return:
        """
        while True:
            ids = list(range(len(self.data)))
            np.random.shuffle(ids)
            X1,X2,Y = [],[],[]
            for i in ids:
                text = self.data[i][0]
                label = self.data[i][1]
                x1,x2 = tokenizer.encode(first=text)
                X1.append(x1)
                X2.append(x2)
                Y.append([label])
                if len(X1)==args.BATCH_SIZE or i == ids[-1]:
                    X1 = data_padding(X1)
                    X2 = data_padding(X2)
                    Y = data_padding(Y)
                    # 多分类问题需要将Y转换为高维向量
                    Y = to_categorical(Y,len(args.LABELS))
                    yield [X1,X2],Y
                    X1,X2,Y = [],[],[]

    def __len__(self):
        return self.step

class Bert4Classify():
    def __init__(self):
        self.labels = args.LABELS

    def load_data(self,):
        """
        导入文本数据
        :return:
        """
        with open(args.DATA_PATH,'r',encoding='utf-8') as f:
            df = f.readlines()
        _data = []
        for _d in df:
            label = self.labels[_d.split('\t')[0].strip()]
            text = ''.join(_d.split('\t')[1:])[:args.MAX_LEN]
            _data.append((text,label))
        print('_data:')
        print(_data[0])
        return _data

    def build_model(self):
        """
        加载bert模型
        :return:
        """
        bert_model = load_trained_model_from_checkpoint(
                                        config_file=args.BERT_CONFIG,
                                        checkpoint_file=args.BERT_MODEL,
                                        trainable=True
                                    )

        input_x1 = Input(shape=(None,))
        input_x2 = Input(shape=(None,))
        x = bert_model([input_x1,input_x2])
        x = Lambda(lambda x:x[:,0])(x)
        x = Dropout(0.5)(x)
        p = Dense(len(args.LABELS),activation='softmax')(x)

        model = Model([input_x1,input_x2],p)
        model.compile(optimizer=Adam(lr=1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def _callbacks(self,):
        """
        回调函数
        :return:
        """
        # 定义回调函数，包括保存模型，提前结束，自动降低学习率
        # 保存模型训练过程
        checkpoint=ModelCheckpoint(filepath=args.MODEL_PATH,monitor='val_loss',verbose=1,save_weights_only=False,save_best_only=True)
        # early_stopping如果3个epoch内validation_loss没有改善则停止训练
        earlystopping=EarlyStopping(monitor='val_loss',patience=3,verbose=1)
        # 自动降低learning_rate
        lr_reduction=ReduceLROnPlateau(monitor='val_loss',factor=0.1,min_lr=1e-5,patience=0,verbose=1)
        csvlog = CSVLogger(filename=args.LOG_PATH)
        callbacks=[earlystopping,lr_reduction,checkpoint,csvlog]
        return callbacks

    def _train(self):
        """
        # 模型训练主调度程序
        :return:
        """
        # 加载数据
        _data = self.load_data()
        # 建模
        model = self.build_model()
        # 回调函数
        callbacks = self._callbacks()
        # KFold
        _data_index = KFold(n_splits=3,shuffle=True).split(_data)
        for train_index,valid_index in _data_index:
            train_data = [_data[_i] for _i in train_index]
            valid_data = [_data[_i] for _i in valid_index]

            # 数据生成器
            train_data_generator = DataGenerator(data=train_data)
            valid_data_generator = DataGenerator(data=valid_data)
            # 训练
            model.fit_generator(
                    train_data_generator.__iter__(),
                    steps_per_epoch=train_data_generator.__len__(),
                    callbacks=callbacks,
                    epochs=args.EPOCH,
                    verbose=1,
                    validation_data=valid_data_generator.__iter__(),
                    validation_steps=valid_data_generator.__len__()
            )

    
if __name__ == '__main__':
    Bert4Classify()._train()

