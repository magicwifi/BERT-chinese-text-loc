# -*- coding:utf-8*-
from __future__ import print_function


import requests
from classifier import *
import time
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

endpoint_title = 'http://127.0.0.1:8500'

endpoint_article = 'http://127.0.0.1:8600'


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)


    """The actual input function."""
    batch_size = FLAGS.predict_batch_size

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

class Client:
    def __init__(self):
        self.processor = MyProcessor()

    def sort_and_retrive(self, predictions, qa_pairs):
        res = []
        for prediction, qa in zip(predictions, qa_pairs):
            res.append((prediction[1], qa))
        res.sort(reverse=True)
        return res

    def preprocess(self, sentences):
        save_path = os.path.join(FLAGS.data_dir, "pred.tsv")
        with open(save_path, 'w') as fout:
            out_line = '0' + '\t' + ' ' + '\n'
            fout.write(out_line)
            for sentence in sentences:
                out_line = '0'+'\t' + sentence + '\n'
                fout.write(out_line)

    def predict_title(self, input_text):
        sentences = [["0",input_text.encode("utf-8")]]
        predict_examples = self.processor.create_examples(sentences, set_type='test',file_base=False)
        label_list = self.processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        features = convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer)

        predict_dataset = input_fn_builder(
            features=features,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        iterator = predict_dataset.make_one_shot_iterator()

        next_element = iterator.get_next()

        inputs = ["label_ids", "input_ids", "input_mask", "segment_ids"]
        for input in inputs:
            next_element[input] = next_element[input].numpy().tolist()

        json_data = {"model_name": "default", "data": next_element}

        start = time.time()
        result = requests.post(endpoint_title, json=json_data)
        cost = time.time() - start
        print('total time cost: %s s' % cost)

        result = dict(result.json())


        output = [np.argmax(i)-1 for i in result['output']]
        if output[0]==0:
            return "利好"
        elif output[0]==1:
            return "利空"
        else:
            return "中性"

        return "中性"

    def predict_article(self, input_text):
        sentences = [["0",input_text.encode("utf-8")]]
        predict_examples = self.processor.create_examples(sentences, set_type='test',file_base=False)
        label_list = self.processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        features = convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer)

        predict_dataset = input_fn_builder(
            features=features,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        iterator = predict_dataset.make_one_shot_iterator()

        next_element = iterator.get_next()

        inputs = ["label_ids", "input_ids", "input_mask", "segment_ids"]
        for input in inputs:
            next_element[input] = next_element[input].numpy().tolist()

        json_data = {"model_name": "default", "data": next_element}

        start = time.time()
        result = requests.post(endpoint_article, json=json_data)
        cost = time.time() - start
        print('total time cost: %s s' % cost)

        result = dict(result.json())

        output = [np.argmax(i)-1 for i in result['output']]
        if output[0]==0:
            return "利好"
        elif output[0]==1:
            return "利空"
        else:
            return "中性"

        return "中性"



from flask import Flask, jsonify
from flask_cors import CORS

import requests
from bs4 import BeautifulSoup



app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
CORS(app)

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}

@app.route('/')
def api_root():
  return "hello"



####


import time
from bert_base.client import BertClient

import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser

args = get_args_parser()

model_dir = r'/Users/zhuangzhuanghuang/Code/output_ner'
bert_dir = '/Users/zhuangzhuanghuang/Code/chinese_L-12_H-768_A-12'


is_training=False
use_one_hot_embeddings=False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids =np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        print(id2label)
        while True:
            print('input the test sentence:')
            sentence = str(input())
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence)
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            print(pred_label_result)
            #todo: 组合策略
            result = strage_combined_link_org_loc(sentence, pred_label_result[0])
            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result



def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        print(', '.join(line))

    def return_output(data):
        line = []
        for i in data:
            line.append(i.word)
        return ', '.join(line)

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    print_output(loc, 'LOC')
    print_output(person, 'PER')
    print_output(org, 'ORG')
    return return_output(person), return_output(loc), return_output(org)


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))





def ner_test():
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
        start_t = time.perf_counter()
        str1 = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
        rst = bc.encode([list(str1)], is_tokenized=True)
        #str1 = list(str1)
        #rst = bc.encode([str1], is_tokenized=True)
        print('rst:', rst)
        print(len(rst[0]))
        person, loc, org = strage_combined_link_org_loc(str1, rst[0])
        print(time.perf_counter() - start_t)


@app.route('/content_title/<path:subpath>')
def api_url(subpath):
  client = Client()
  #article ='';
  title = ''
  try:
    html = requests.get(subpath, headers = headers,);
    #html.encoding='utf-8'
    print(subpath)
    bs = BeautifulSoup(html.text, "html.parser")

    title_node = bs.find('title')
    title =  title_node.get_text()
    print(title)

  except Exception:
    #article ='';
    title = ''


  with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    rst = bc.encode(list(",".join(title.split())))
    person, loc, org = strage_combined_link_org_loc(title, rst[0])
    prediction = client.predict_title(title)
    print(prediction)

  return jsonify({'prediction':prediction,'title':title,'person':person,'loc':loc,'org':org})

@app.route('/content_article/<path:subpath>')
def api_article(subpath):
  client = Client()
  article ='';
  #title = ''
  try:
    html = requests.get(subpath, headers = headers,);
    #html.encoding='utf-8'
    print(subpath)
    bs = BeautifulSoup(html.text, "html.parser")
    part = bs.find_all('p')
    if len(part)>20:
      for paragraph in part[:20]:
        article += str(paragraph.get_text().strip())
    else:
      for paragraph in part:
        article += str(paragraph.get_text().strip())
    print(article)

  except Exception:
    article ='';
    #title = ''
  prediction = client.predict_article(article)
  print(prediction)



  return jsonify({'prediction':prediction,'article':article})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
