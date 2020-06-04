import os,sys

class Config():

    bert_dir = os.path.join(sys.path[0],'bert','chinese_L-12_H-768_A-12')
    bert_config = os.path.join(sys.path[0],'bert','chinese_L-12_H-768_A-12','bert_config.json')
    bert_model = os.path.join(sys.path[0],'bert','chinese_L-12_H-768_A-12','bert_model.ckpt')
    vocab = os.path.join(sys.path[0],'bert','chinese_L-12_H-768_A-12','vocab.txt')

    data_file = os.path.join(sys.path[0],'data','cnews.train.txt')
    labels = {'时尚': 0, '科技': 1, '娱乐': 2, '房产': 3, '财经': 4, '游戏': 5, '教育': 6, '家居': 7, '体育': 8, '时政': 9}
    max_text_length = 68

    model_path = os.path.join(sys.path[0],'model','model.h5')
    log_path = os.path.join(sys.path[0],'log','train.log')

    batchsize = 32
    epoch = 20

if __name__ == '__main__':
    print(Config().labels.__len__())