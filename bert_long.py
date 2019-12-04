# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import collections
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import pandas as pd
import pickle
import unicodedata
from sklearn.metrics import accuracy_score, f1_score
import random

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
base='./bert_quora'
## Required parameters
# if os.name == 'nt':
#     bert_path = 'D:\\Research\\multi_cased_L-12_H-768_A-12'
#     root_path = r'D:/Research/Bert_Ner'
# else:
#     # bert_path = '/home/linhlt/matt/bert_ner/bert-models/multi_cased_L-12_H-768_A-12'
#     # root_path = '/home/linhlt/Levi/chatbot_platform_nlp'
bert_path = 'gs://test_bucket_share_1/uncased_L-12_H-768_A-12'
project_path='./drive/My Drive/AI_COLAB/BERT_tensor'
root_path = base

flags.DEFINE_string(
    "data_dir", os.path.join(project_path, 'data_train_new'),
    "The input datadir.",
)

flags.DEFINE_string(
    "test_dir", os.path.join(project_path, 'data_test_new'),
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", 'Uland', "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir",'gs://test_bucket_share_1/model_trained/new_model_312',
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name",'grpc://10.61.3.106:8470' ,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-3, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 50.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string('data_config_path', os.path.join(root_path, 'data.conf'),
                    'data config file, which save train and dev config')


# from bert_classification import ai_const_production as ai_const
import glob
import json
# import unidecode


# def get_data_dictionary_from_excel_linhlt(excel_file):
#     print("READING_FILE", excel_file)
#     xl = pd.ExcelFile(excel_file)
#     df = xl.parse('SE result')
#     df.dropna(subset=["Sentence"], inplace=True)
#     folder1s = df["Folder1"].values
#     folder2s = df["Folder2"].values
#     files = df["File"].values
#     tags = df["Tag"].values
#     tags = [unicodedata.normalize('NFKC', re.sub('＊|\*|\s+', '', x)).lower() if type(x) == str else x for x in tags]
#     # tags = [re.sub('\s+','',x) for x in tags]
#     is_titles = df["Is_title"].values
#     is_tables = df["Is_table"].values
#     values = df["Value"].values
#     sentences = df["Sentence"].values
#     all_titles = df["Titles"].values
#     is_tag11_done=False
#     for i, x in enumerate(all_titles):
#         try:
#             if type(x)==str:
#                 x = re.sub(',\s*\]', ']', x)
#                 x = re.sub(';\"', '\"', x)
#                 x = re.sub(',\"\s*\"','","',x)
#                 x = re.sub(',\s*,', ',', x)
#
#                 all_titles[i] = json.loads(x, strict=False)
#             else:
#                 all_titles[i] =[]
#         except Exception:
#             # all_titles[i] = []
#             print("line_err", i, x)
#             raise
#     X_sent=[]
#     X_title=[]
#     Y=[]
#     for i, sentence in enumerate(sentences):
#         # print("LINE_NUMBER",i)
#         # if i == 16 or i == 17:
#         #     print("DEBUG")
#         titles = all_titles[i]
#         current_tags = tags[i].split(';') if type(tags[i]) == str else []
#         current_tags = [x for x in current_tags if len(re.sub('\s+', '', x)) > 0]
#         # current_in_tags = [x for x in current_tags if x in ai_const.all_tags]
#         # current_not_in_tags= [x for x in current_tags if x not in current_in_tags]
#         # current_not_in_tags= random.sample( current_not_in_tags, 2* len(current_in_tags))
#         # current_tags= current_in_tags+ current_not_in_tags
#         if len(current_tags)==0:
#             continue
#         # curr_sentence="".join(titles)+sentence
#         X_sent.append(sentence)
#         if len(titles)>0:
#             X_title.append("".join(titles))
#         else:
#             X_title.append(sentence)
#         Y.append(current_tags)
#     print("LEN TRAINING: ", len(X_sent))
#     return X_sent, X_title, Y

# def get_data_dictionary_from_folder_linhlt(excel_folder):
#     X_sent=[]
#     X_title=[]
#     Y=[]
#     excel_file_paths = glob.glob(excel_folder + '/**/*.xlsx', recursive=True)
#     for i,excel_file in enumerate(excel_file_paths):
#         if '.xlsx.xlsx' not in excel_file:
#             try:
#                 x_sent, x_title, y=get_data_dictionary_from_excel_linhlt(excel_file)
#                 X_sent.extend(x_sent)
#                 X_title.extend(x_title)
#                 Y.extend(y)
#             except:
#                 print('ERROR FILE: ', excel_file)
#     return X_sent, X_title, Y

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a,text_b, labels=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.labels = labels



class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  # def get_dev_examples(self, data_dir):
  #   """Gets a collection of `InputExample`s for the dev set."""
  #   raise NotImplementedError()
  #
  # def get_test_examples(self, data_dir):
  #   """Gets a collection of `InputExample`s for prediction."""
  #   raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_csv(cls, data_dir):
    """Reads a tab separated value file."""
    y = []
    X1 = []
    X2 = []
    df = pd.read_csv(data_dir, sep='\t', encoding='utf-8', error_bad_lines=False)
    for i in df.index:
        try:
            y.append(str(int(df['is_duplicate'][i])))
            X1.append(str(df['question1'][i]))
            X2.append(str(df['question2'][i]))
        except:
            pass
    return (X1, X2, y)

  # def read_kepco_multilabel(self,excel_folder):
  #     return get_data_dictionary_from_folder_linhlt(excel_folder)



class UlandProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        (X_sent,X_title, Y) = self._read_csv(os.path.join(data_dir,'train.tsv'))
        # Y=[[tag for tag in y if tag in ai_const.all_tags] for y in Y]
        # print('LEN NO TAG: ', )
        examples = []
        for i in range(len(X_sent)):
            # if i == 0:
            #     continue
            guid = "train-%d" % (i)
            if not isinstance(X_sent[i], str):
                continue
            text_a = tokenization.convert_to_unicode(X_sent[i])
            text_b = tokenization.convert_to_unicode(X_title[i])
            labels = tokenization.convert_to_unicode(Y[i])
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b= text_b, labels=labels))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        (X1, X2, Y) = self._read_csv(os.path.join(data_dir, 'test.tsv'))
        num_yes = sum([1 if y == 'true' else 0 for y in Y])
        print("------------------------------------------------")
        print('NUM Yes: ', num_yes)
        print("------------------------------------------------")
        examples = []
        for i in range(len(X1)):
            # if i == 0:
            #     continue
            guid = "test-%d" % (i)
            if not isinstance(X1[i], str):
                continue
            text1 = tokenization.convert_to_unicode(X1[i])
            text2 = tokenization.convert_to_unicode(X2[i])
            label = tokenization.convert_to_unicode(Y[i])
            examples.append(
                InputExample(guid=guid, text_a=text1, text_b=text2, labels=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        (X1, X2, Y) = self._read_csv(os.path.join(data_dir, 'dev.tsv'))
        num_yes = sum([1 if y == 'true' else 0 for y in Y])
        print("------------------------------------------------")
        print('NUM Yes: ', num_yes)
        print("------------------------------------------------")
        examples = []
        for i in range(len(X1)):
            # if i == 0:
            #     continue
            guid = "dev-%d" % (i)
            if not isinstance(X1[i], str):
                continue
            text1 = tokenization.convert_to_unicode(X1[i])
            text2 = tokenization.convert_to_unicode(X2[i])
            label = tokenization.convert_to_unicode(Y[i])
            examples.append(
                InputExample(guid=guid, text_a=text1, text_b=text2, labels=label))
        return examples

    # def get_train_examples(self, data_dir):
    #     X, Y= get_data_dictionary_from_folder_linhlt(data_dir)


    def get_labels(self):
        # return [label.lower() for label in  ai_const.all_tags]
        return ['1', '0']

def convert_to_k_hot(labels, num_labels):
    # print("NUM LABEL: ", num_labels)
    labels_=[]
    for i in range(num_labels):
        if i in labels:
            labels_.append(1)
        else:
            labels_.append(0)
    return labels_

def convert_single_example( example:InputExample, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  # print('LABEL _ LIST: ', label_list)
  # print('LABEL _ MAP : ', label_map)
  with open(os.path.join(FLAGS.data_dir, 'label2id.pkl'), 'wb') as w:
      pickle.dump(label_map, w)

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = tokenizer.tokenize(example.text_b)

  if len(tokens_a) > int( max_seq_length/2 - 1):
    tokens_a = tokens_a[0:int( max_seq_length/2 - 1)]
  if len(tokens_b) > int( max_seq_length/2 - 1):
    tokens_b = tokens_b[0:int( max_seq_length/2 - 1)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  tokens.append("[CLS]")
  segment_ids.append(1)
  #TODO: For sentence b
  for token in tokens_b:
    tokens.append(token)
    segment_ids.append(1)
  tokens.append("[SEP]")
  segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  label_ids = label_map[example.labels]
  # try:
  #   label_ids = [label_map[label] for label in example.labels]
  #   label_ids = convert_to_k_hot(label_ids, len(label_map))
  # except:
  #   print(label_map)
  #   1/0
  # if ex_index < 5:
  #   tf.logging.info("*** Example ***")
  #   tf.logging.info("guid: %s" % (example.guid))
  #   tf.logging.info("tokens: %s" % " ".join(
  #       [tokenization.printable_text(x) for x in tokens]))
  #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
  #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
  #   tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
  #   tf.logging.info("label: %s (id = %d)" % (example.labels, label_ids))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_ids)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example( example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_list_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_ids])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    print(example)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  #TODO: Get position of second sentence


  with tf.variable_scope("finetune"):
    sequence_output= model.get_sequence_output()
    shape= sequence_output.shape
    batch_size= shape[0]
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    # segment_ids = tf.Print(segment_ids, [segment_ids], )
    #TODO: get index of 2nd sentence
    second_sentence_pos = tf.cast( tf.argmax(segment_ids, axis=1), tf.int32)
    # second_sentence_pos= tf.Print(second_sentence_pos,[second_sentence_pos], summarize=10)
    # second_sentence_pos = tf.Print(second_sentence_pos, [second_sentence_pos], summarize= 10)
    rows= tf.range(batch_size)
    # rows= tf.Print(rows, [rows])
    second_sentence_pos= tf.stack([rows, second_sentence_pos], axis=1)
    # second_sentence_pos= tf.Print(second_sentence_pos, [second_sentence_pos], summarize= 10)
    second_token_tensor = tf.gather_nd(params=sequence_output, indices=second_sentence_pos)
    print('FIRST TOKEN TENSOR SHAPE: ', first_token_tensor.shape)
    print("SECOND TOKEN TENSOR SHAPE: ", second_token_tensor.shape)
    # second_token_tensor = tf.squeeze(sequence_output[:, second_sentence_pos:second_sentence_pos+1, :], axis=1)
    first_ebeding= tf.layers.dense(
            first_token_tensor,
            bert_config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(bert_config.initializer_range))
    second_ebeding= tf.layers.dense(
            second_token_tensor,
            bert_config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(bert_config.initializer_range))
    # second_ebeding= tf.Print(second_ebeding, [second_ebeding])?

    output_layer= tf.concat(values=[first_ebeding, second_ebeding], axis=1)

  hidden_size = 2* bert_config.hidden_size#output_layer.shape[-1].value

  with tf.variable_scope("sent_a"):
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())



  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    # logits = tf.sigmoid(logits)
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    # one_hot_labels=tf.cast(labels, tf.float32)  #TODO: convert to k-hot
    # print('Label shape: ',one_hot_labels.shape)
    # print('NUM LABEL: ', num_labels)
    # tf.reshape(one_hot_labels, [-1, num_labels])
    # one_hot_labels= [tf.size(one_hot_labels)]
    # one_hot_labels= tf.Print(one_hot_labels,[one_hot_labels], message='label', summarize=10)
    # logits = tf.Print(logits, [logits], message='logits', summarize=10)

    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
    # loss = tf.reduce_sum(loss)
    # loss = tf.Print(loss, [loss], message='loss')
    # return (loss, None, logits, logits)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)
    loss= tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
    loss = tf.reduce_mean(loss)
    loss = tf.Print(loss, [loss], message='loss')
    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    print('Label ids shape: ',label_ids.shape)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
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

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

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

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example( example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "uland": UlandProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))



  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))



  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  #-------------------------------------------------------------------------------------------------------------------
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))


  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tf.gfile.MakeDirs(FLAGS.output_dir)
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


import re
class BertMultilabelClassifier(object):
    def __init__(self):
        tf.reset_default_graph()
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        init_checkpoint = FLAGS.init_checkpoint
        self.use_one_hot_embeddings = False
        self.processor= UlandProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels= len(self.label_list)
        self.predict = self._build(bert_config, self.use_one_hot_embeddings)
        lastest_checkpoint=tf.train.latest_checkpoint(FLAGS.output_dir)
        self.initialize_and_restore_session(lastest_checkpoint)
        self.eval_features = []
        self.ori_eval_features = []
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        with open(os.path.join(FLAGS.data_dir, 'label2id.pkl'), 'rb') as w:
            label_map= pickle.load(w)
        self.label_map = dict((v, k) for k, v in label_map.items())
        pass

    def initialize_and_restore_session(self, model_dir):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, device_count={'GPU': 0})
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_dir)
        print("Complete restore model from ", model_dir)

    def create_model(self,bert_config, is_training, input_ids, input_mask, segment_ids, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        # TODO: Get position of second sentence

        with tf.variable_scope("finetune"):
            sequence_output = model.get_sequence_output()
            shape = sequence_output.shape
            print('SHAPE 1: ', shape)
            # 1/0
            # batch_size = shape[0]
            first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
            # first_token_tensor = tf.Print(first_token_tensor, [first_token_tensor], message='first', summarize=10)
            print('FIRST TENSOR SHAPE: ', first_token_tensor.shape)
            # segment_ids = tf.Print(segment_ids, [segment_ids], )
            # TODO: get index of 2nd sentence
            second_sentence_pos = tf.cast(tf.argmax(segment_ids, axis=1), tf.int32)

            rows = tf.constant([0])
            second_sentence_pos = tf.stack([rows, second_sentence_pos], axis=1)
            # second_sentence_pos= tf.Print(second_sentence_pos, [second_sentence_pos])
            second_token_tensor = tf.gather_nd(params=sequence_output, indices=second_sentence_pos)
            # second_token_tensor= tf.Print(second_token_tensor,[second_token_tensor],  message='second', summarize=10)
            print('SECOND TENSOR SHAPE: ', second_token_tensor.shape)
            # second_token_tensor = tf.squeeze(sequence_output[:, second_sentence_pos:second_sentence_pos+1, :], axis=1)
            first_ebeding = tf.layers.dense(
                first_token_tensor,
                bert_config.hidden_size,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(bert_config.initializer_range))
            second_ebeding = tf.layers.dense(
                second_token_tensor,
                bert_config.hidden_size,
                activation=tf.tanh,
                kernel_initializer=tf.truncated_normal_initializer(bert_config.initializer_range))
            # second_ebeding= tf.Print(second_ebeding, [second_ebeding])?

            output_layer = tf.concat(values=[first_ebeding, second_ebeding], axis=1)

        hidden_size = 2 * bert_config.hidden_size  # output_layer.shape[-1].value

        with tf.variable_scope("sent_a"):
            output_weights = tf.get_variable(
                "output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            # logits = tf.sigmoid(logits)
            # probabilities = tf.nn.softmax(logits, axis=-1)
            # log_probs = tf.nn.log_softmax(logits, axis=-1)

            # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            # one_hot_labels = tf.cast(labels, tf.float32)  # TODO: convert to k-hot
            # print('Label shape: ', one_hot_labels.shape)
            # print('NUM LABEL: ', num_labels)
            # # tf.reshape(one_hot_labels, [-1, num_labels])
            # # one_hot_labels= [tf.size(one_hot_labels)]
            #
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
            # loss = tf.reduce_sum(loss)
            # loss = tf.Print(loss, [loss])

            return (0, None, logits, logits)

    def _build(self, bert_config, use_one_hot_embeddings):
        self.input_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.input_mask = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        (_,_,logits,_) = self.create_model(bert_config, False, self.input_ids, self.input_mask,
                 self.segment_ids, self.num_labels, use_one_hot_embeddings)
        return logits

    def init_checkpoint(self, init_checkpoint):
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        pass

    def get_feed_dict(self, features):
        feed={}
        feed[self.input_ids]=features['input_ids']
        feed[self.input_mask]= features['input_mask']
        feed[self.segment_ids]= features['segment_ids']
        return feed


    def predict_raw_sentence(self,example):
        fetures=self.convert_single_example_to_features(example,
                                                 FLAGS.max_seq_length, self.tokenizer)
        logits= self.sess.run([self.predict],feed_dict=self.get_feed_dict(fetures))
        logits_=logits[0][0]
        # logits= tf.reshape(logits, [28])
        logits =[]
        logits.append(logits_)
        logits.append(logits_)
        logits=np.asarray(logits)
        zeros=[0 for i in range(len(logits_))]
        ones =[1 for i in range(len(logits_))]
        mark=[]
        mark.append(zeros)
        mark.append(ones)
        mark= np.asarray(mark)
        pred= np.argmin(np.abs(logits-mark), axis=0)
        return pred

    def get_test_example(self, folder_dir):
        return self.processor.get_test_examples(folder_dir)

    def convert_single_prediction_example(self,example, max_seq_length, tokenizer):
        label_list = self.label_list
        return convert_single_example(example, label_list, max_seq_length,tokenizer)

    def convert_single_example_to_features(self, example, max_seq_length, tokenizer):
        feature = self.convert_single_prediction_example(example, max_seq_length, tokenizer)
        def create_int_feature(values):
            f = list(values)
            return f
        features = collections.OrderedDict()
        features["input_ids"] = [create_int_feature(feature.input_ids)]
        features["input_mask"] = [create_int_feature(feature.input_mask)]
        features["segment_ids"] = [create_int_feature(feature.segment_ids)]
        return features

    def test(self,test_dir):
        examples= self.get_test_example(test_dir)
        Y_true=[]
        Y_test=[]
        label_map = {}
        for (i, label) in enumerate(self.label_list):
            label_map[label] = i
        zero=0
        for example in examples:
            # label= example.labels
            label_ids = [label_map[example.labels]]
            # for label in example.labels
            if len(label_ids)==0:
                zero+=1
            label = convert_to_k_hot(label_ids, len(label_map))
            # print('L: ',label)
            Y_true.append(label)
            Y_test.append( self.predict_raw_sentence(example))
        print('ZERO: ', zero)
        # 1/0
        Y_true= np.asarray(Y_true)
        Y_test = np.asarray(Y_test)
        true_dict={i:[] for i in range(len(Y_true[0]))}
        pred_dict={i:[] for i in range(len(Y_true[0]))}
        for i in range(len(Y_true)):
            for j in range(len(true_dict)):
                true_dict[j].append(Y_true[i][j])
                pred_dict[j].append(Y_test[i][j])
        for i in range(len(Y_true[0])):
            print('F1 ', i, ' : ', f1_score(true_dict[i], pred_dict[i]))
        # for i in range(len(Y_true)):
        #     if list(Y_true[i])== list(Y_test[i]):
        #         print(Y_test[i])
        #         print(i)
        print('ACC: ',accuracy_score(Y_true, Y_test))


def run():
    main()
    cls= BertMultilabelClassifier()
    cls.test(FLAGS.test_dir)
run()