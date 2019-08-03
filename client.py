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
        print(output[0])
        if output[0]==0:
            return "利好"
        elif output[0]==1:
            return "利空"
        else:
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
        print(output[0])
        if output[0]=="0":
            return "利好"
        elif output[0]=="1":
            return "利空"
        else:
            return "中性"


if __name__ == '__main__':
    client = Client()
    #msg = ["港交所高层涉贪 30家企业被放水 小米、美团等曾遭首日破发","地产观察：碧桂园用靓丽业绩粉碎流动性传言","港股全线爆发：波动中凸显价值 大幅反弹原因揭秘"]

    while True:
        print('input the test sentence:')
        sentence = str(input())

        print(len(sentence))
        if len(sentence) < 2:
            print(sentence)
            continue
        prediction = client.predict_title(sentence)
    # print('probability: %s'%prediction)
        print(prediction)

        #prediction = client.predict_title(sentence)
    # print('probability: %s'%prediction)
        #print(prediction)
    #prediction = client.predict_article(msg)
    # print('probability: %s'%prediction)
    #print(prediction)

    #prediction = client.predict_title(msg)
    #print('probability: %s'%prediction)
    #print(prediction)
    #prediction = client.predict_article(msg)
    # print('probability: %s'%prediction)
    #print(prediction)


