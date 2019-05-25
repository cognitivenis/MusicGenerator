#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:31:10 2019

@author: djordjepav
"""

import os
import pretty_midi
import pickle

import numpy as np
import tensorflow as tf

from numpy.random import choice
from seq_self_attention import SeqSelfAttention

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)

    return pm


def generate_from_random(unique_notes, seq_len=50):
    generate = np.random.randint(0,unique_notes,seq_len).tolist()
    return generate


def generate_from_one_note(note_tokenizer, new_notes='35', seq_len=50):
    generate = [note_tokenizer.notes_to_index['e'] for i in range(seq_len-1)]
    generate += [note_tokenizer.notes_to_index[new_notes]]
    return generate


def generate_from_notes(note_tokenizer, notes=['35'], seq_len=50):
    generate = [note_tokenizer.notes_to_index['e'] for i in range(seq_len-len(notes))]
    generate += [note_tokenizer.notes_to_index[note] for note in notes]
    return generate


def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len=50):
    for i in range(max_generated):
        test_input = np.array([generate])[:,i:i+seq_len]
        predicted_note = model.predict(test_input)
        random_note_pred = choice(unique_notes+1, 1, replace=False, p=predicted_note[0])
        generate.append(random_note_pred[0])

    return generate


def write_midi_file_from_generated(generate, midi_file_name = "generated_song.mid", start_index=49,
                                   fs=8, max_generated=1000):
    note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    array_piano_roll = np.zeros((128,max_generated+1), dtype=np.int16)
    for index, note in enumerate(note_string[start_index:]):
        if note == 'e':
            pass
        else:
            splitted_note = note.split(',')
            for j in splitted_note:
                array_piano_roll[int(j),index] = 1
    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    print("Tempo {}".format(generate_to_midi.estimate_tempo()))
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = np.random.randint(50, 100)
    generate_to_midi.write(os.path.join('../generated_songs/',midi_file_name))


#===================================================================================================


DIR = '../models/Johann Sebastian Bach/'
MODEL = 'epoch10_v2.h5'
TYPE = 'random'

model = tf.keras.models.load_model(os.path.join(DIR, MODEL),
                                   custom_objects=SeqSelfAttention.get_custom_objects())
note_tokenizer = pickle.load(open(os.path.join(DIR, "tokenizer.p"), "rb"))

if TYPE == 'random':
    max_generate = 500
    unique_notes = note_tokenizer.unique_word
    seq_len = 100
    generate = generate_from_random(unique_notes, seq_len)
    generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
    write_midi_file_from_generated(generate, "random.mid", start_index=seq_len-1, fs=6,
                                   max_generated=max_generate)
elif TYPE == 'notes':
    max_generate = 500
    unique_notes = note_tokenizer.unique_word
    seq_len = 50
    generate = generate_from_notes(note_tokenizer, ['38', '48', '45', '48', '50'])
    generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
    new_midi = write_midi_file_from_generated(generate, "new_song.mid", start_index=seq_len-1, fs=7,
                                   max_generated=max_generate)
else :
    max_generate = 500
    unique_notes = note_tokenizer.unique_word
    seq_len = 50
    generate = generate_from_one_note(note_tokenizer, '38')
    generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
    write_midi_file_from_generated(generate, "one_note.mid", start_index=seq_len-1, fs=7,
                                   max_generated=max_generate)
