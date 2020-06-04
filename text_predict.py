# coding:utf-8
import numpy as np
from keras.optimizers import Adam
from keras_bert import load_vocabulary,Tokenizer
from config import Config
import argparse
from keras.models import load_model
from keras_bert import get_custom_objects

CONFIG = Config()
parse = argparse.ArgumentParser()
parse.add_argument('-v','--VOCAB',default=CONFIG.vocab,help='bert词汇表')
parse.add_argument('-m','--MODEL_PATH',default=CONFIG.bert_model,help='模型保存路径')
parse.add_argument('-l','--LABELS',default=CONFIG.labels,help='文本总的类别')
args = parse.parse_args()

# 加载bert分词器
token_dict = load_vocabulary(vocab_path=args.VOCAB)
tokenizer = Tokenizer(token_dict=token_dict)
def _id2label():
    id2label = {}
    for k,v in args.LABELS.items():
        id2label[v] = k
    return id2label


# 加载模型
model = load_model(args.MODEL_PATH,custom_objects=get_custom_objects())
id2label = _id2label()

def _predict(**data):
    """
    # 模型预测
    :return:
    """
    text = data['text']
    x1,x2 = tokenizer.encode(first=text)
    a = id2label[np.argmax(model.predict([x1,x2]))]
    print(a)
    
if __name__ == '__main__':
    _predict(text='美国全国发生暴乱')