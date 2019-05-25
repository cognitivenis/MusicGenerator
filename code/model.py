#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:31:00 2019

@author: djordjepav
"""

import os
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, Embedding, LSTM, Dropout, Dense, LeakyReLU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from random import shuffle

from seq_self_attention import SeqSelfAttention
from preprocessing import NoteTokenizer
from preprocessing import generate_batch_song, generate_dict_time_notes, process_notes_in_song, get_sampled_midi


def get_model(seq_len, unique_notes, dropout=0.3, output_embedding=100, lstm_unit=128,
              dense_unit=64):

    inputs = Input(shape=(seq_len,), name='input')
    x = Embedding(input_dim=unique_notes+1, output_dim=output_embedding,
                  input_length=seq_len, name='embedding')(inputs)
    x = Bidirectional(LSTM(lstm_unit, return_sequences=True), name='lstm_1')(x)
    x , _ = SeqSelfAttention(return_attention=True,
                             attention_activation='sigmoid',
                             attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                             attention_width=100,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                             bias_regularizer=tf.keras.regularizers.l1(1e-4),
                             attention_regularizer_weight=1e-4,
                             name='self_attention_1')(x)
    x = Dropout(dropout, name='dropout_1')(x)
    x = Bidirectional(LSTM(lstm_unit, return_sequences=True), name='lstm_2')(x)
    x , _ = SeqSelfAttention(return_attention=True,
                             attention_activation='sigmoid',
                             attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                             attention_width=100,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                             bias_regularizer=tf.keras.regularizers.l1(1e-4),
                             attention_regularizer_weight=1e-4,
                             name='self_attention_2')(x)
    x = Dropout(dropout, name='dropout_2')(x)
    x = Bidirectional(LSTM(lstm_unit), name='lstm_3')(x)
    x = Dropout(dropout, name='dropout_3')(x)
    x = Dense(dense_unit, name='dense_1')(x)
    x = LeakyReLU(name='leaky_re_lu')(x)
    outputs = Dense(unique_notes+1, activation = "softmax", name='dense_2')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


class TrainModel:

    def __init__(self, epochs, note_tokenizer, sampled_midi, fs, batch_size, batch_song, optimizer,
                 checkpoint, loss_fn, checkpoint_prefix, total_songs, model, seq_len, directory):

        self.epochs = epochs
        self.note_tokenizer = note_tokenizer
        self.sampled_midi = sampled_midi
        self.fs = fs
        self.batch_size = batch_size
        self.batch_song = batch_song
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.loss_fn = loss_fn
        self.checkpoint_prefix = checkpoint_prefix
        self.total_songs = total_songs
        self.model = model
        self.seq_len = seq_len
        self.directory = directory

    def train(self):
        for epoch in range(self.epochs):
            shuffle(self.sampled_midi)
            loss_total = 0
            steps = 0
            steps_nnet = 0

            for i in range(0, self.total_songs, self.batch_song):
                steps += 1
                inputs_large, outputs_large = generate_batch_song(list_all_midi=self.sampled_midi,
                                                                  batch_music=self.batch_song,
                                                                  start_index=i, fs=self.fs,
                                                                  seq_len=self.seq_len)
                inputs_large = np.array(self.note_tokenizer.transform(inputs_large), dtype=np.int32)
                outputs_large = np.array(self.note_tokenizer.transform(outputs_large), dtype=np.int32)

                index_shuffled = np.arange(start=0, stop=len(inputs_large))
                np.random.shuffle(index_shuffled)

                for nnet_steps in range(0, len(index_shuffled), self.batch_size):
                    steps_nnet += 1
                    current_index = index_shuffled[nnet_steps:nnet_steps+self.batch_size]
                    inputs, outputs = inputs_large[current_index], outputs_large[current_index]

                    if len(inputs) // self.batch_size != 1:
                        break

                    loss = self.train_step(inputs, outputs)
                    loss_total += tf.math.reduce_sum(loss)

                    if steps_nnet % 50 == 0:
                        print("\n epochs: {} | steps: {} | total_loss={}".format(epoch+1, steps_nnet,
                              loss_total))

            checkpoint.save(file_prefix = self.checkpoint_prefix)
            model.save(os.path.join(self.directory,'epoch{}_v2.h5'.format(epoch+1)))
            pickle.dump(note_tokenizer, open(os.path.join(self.directory, "tokenizer.p"), "wb"))

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss_fn(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


#===================================================================================================


EPOCHS = 10
SEQ_LEN = 100
FS = 5
BATCH_SIZE = 96
BATCH_SONG = 16
DIR = '../models/Johann Sebastian Bach/'

composer = 'Johann Sebastian Bach'

sampled_midi = get_sampled_midi(composer=composer)

note_tokenizer = NoteTokenizer()

for i in range(len(sampled_midi)):
    dict_time_notes = generate_dict_time_notes(sampled_midi, batch_song=1, start_index=i, fs=FS)
    full_notes = process_notes_in_song(dict_time_notes)
    for note in full_notes:
        note_tokenizer.partial_fit(list(note.values()))
note_tokenizer.add_new_note('e')


model = get_model(SEQ_LEN, note_tokenizer.unique_word)
model.summary()

#model.load_weights(os.path.join(DIR,'epoch4.h5'))

optimizer = Nadam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_dir = '.././training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
loss_fn = sparse_categorical_crossentropy

train_model = TrainModel(epochs=EPOCHS, note_tokenizer=note_tokenizer, sampled_midi=sampled_midi,
                         fs=FS, batch_size=BATCH_SIZE, batch_song=BATCH_SONG,
                         optimizer=optimizer, checkpoint=checkpoint, loss_fn=loss_fn,
                         checkpoint_prefix=checkpoint_prefix, total_songs=len(sampled_midi),
                         model=model, seq_len=SEQ_LEN, directory=DIR)
train_model.train()
