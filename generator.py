'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Generate jazz using a deep learning model (LSTM in deepjazz).

Some code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml 
with express permission.

Code was built while significantly referencing public examples from the
Keras documentation on GitHub:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generator.py [# of epochs]

    Note: running Keras/Theano on GPU is formally supported for only NVIDIA cards (CUDA backend).
'''
from __future__ import print_function

import argparse
import sys

from music21 import *
import numpy as np

from grammar import *
from preprocess import *
from qa import *
import lstm

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to sample an index from a probability array '''
def __sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

''' Helper function to generate a predicted value from a given matrix '''
def __predict(model, x, indices_val, diversity):
    preds = model.predict(x, verbose=0)[0]
    next_index = __sample(preds, diversity)
    next_val = indices_val[next_index]

    return next_val

''' Helper function which uses the given model to generate a grammar sequence 
    from a given corpus, indices_val (mapping), abstract_grammars (list), 
    and diversity floating point value. '''
def __generate_grammar(model, corpus, abstract_grammars, values, val_indices,
                       indices_val, max_len, max_tries, diversity):
    curr_grammar = ''
    # np.random.randint is exclusive to high
    start_index = np.random.randint(0, len(corpus) - max_len)
    sentence = corpus[start_index: start_index + max_len]    # seed
    running_length = 0.0
    while running_length <= 4.1:    # arbitrary, from avg in input file
        # transform sentence (previous sequence) to matrix
        x = np.zeros((1, max_len, len(values)))
        for t, val in enumerate(sentence):
            if (not val in val_indices): print(val)
            x[0, t, val_indices[val]] = 1.

        next_val = __predict(model, x, indices_val, diversity)

        # fix first note: must not have < > and not be a rest
        if (running_length < 0.00001):
            tries = 0
            while (next_val.split(',')[0] == 'R' or 
                len(next_val.split(',')) != 2):
                # give up after 1000 tries; random from input's first notes
                if tries >= max_tries:
                    print('Gave up on first note generation after', max_tries, 
                        'tries')
                    # np.random is exclusive to high
                    rand = np.random.randint(0, len(abstract_grammars))
                    next_val = abstract_grammars[rand].split(' ')[0]
                else:
                    next_val = __predict(model, x, indices_val, diversity)

                tries += 1

        # shift sentence over with new value
        sentence = sentence[1:] 
        sentence.append(next_val)

        # except for first case, add a ' ' separator
        if (running_length > 0.00001): curr_grammar += ' '
        curr_grammar += next_val

        length = float(next_val.split(',')[1])
        running_length += length

    return curr_grammar

#----------------------------PUBLIC FUNCTIONS----------------------------------#
''' Generates musical sequence based on the given data filename and settings.
    Plays then stores (MIDI file) the generated output. '''
def generate(data_fn, out_fn, n_epochs, args):

    # model settings
    max_len = 20
    max_tries = 1000
    diversity = 0.5

    # musical settings
    bpm = 100

    # get data
    chords, abstract_grammars = get_musical_data(data_fn, args)
    corpus, values, val_indices, indices_val = get_corpus_data(abstract_grammars)
    print('corpus length:', len(corpus))
    print('total # of values:', len(values))

    # build model
    model = lstm.build_model(corpus=corpus, val_indices=val_indices,
                             max_len=max_len, n_epochs=n_epochs)

    # set up audio stream
    out_stream = stream.Stream()

    # generation loop
    curr_offset = 0.0
    loop_end = len(chords)
    for loop_index in range(1, loop_end):
        # get chords from file
        curr_chords = stream.Voice()
        for j in chords[loop_index]:
            curr_chords.insert((j.offset % 4), j)

        # generate grammar
        curr_grammar = __generate_grammar(model=model, corpus=corpus, 
                                          abstract_grammars=abstract_grammars, 
                                          values=values, val_indices=val_indices, 
                                          indices_val=indices_val, 
                                          max_len=max_len, max_tries=max_tries,
                                          diversity=diversity)

        curr_grammar = curr_grammar.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        curr_grammar = prune_grammar(curr_grammar)

        # Get notes from grammar and chords
        curr_notes = unparse_grammar(curr_grammar, curr_chords)

        # Pruning #2: removing repeated and too close together notes
        curr_notes = prune_notes(curr_notes)

        # quality assurance: clean up notes
        curr_notes = clean_up_notes(curr_notes)

        # print # of notes in curr_notes
        print('After pruning: %s notes' % (len([i for i in curr_notes
            if isinstance(i, note.Note)])))

        # insert into the output stream
        for m in curr_notes:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0

    out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))

    # Play the final stream through output (see 'play' lambda function above)
    play = lambda x: midi.realtime.StreamPlayer(x).play()
    play(out_stream)

    # save stream
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open(out_fn, 'wb')
    mf.write()
    mf.close()

''' Runs generate() -- generating, playing, then storing a musical sequence --
    with the default Metheny file. '''
def main(args):
    n_epochs = args.n

    # i/o settings
    midi_in = args.midi # Default is 'And Then I Knew' by Pat Metheny
    midi_out = 'dj_' + midi_in[:-4]

    data_fn = 'midi/' + midi_in
    out_fn = 'midi/' + midi_out + "_{}-{}_s{}_".format(args.start, args.end, args.splice) + str(n_epochs)
    if (n_epochs == 1): out_fn += '_epoch.midi'
    else:               out_fn += '_epochs.midi'

    generate(data_fn, out_fn, n_epochs, args)

p = argparse.ArgumentParser()

p.add_argument('--n', help="Number of epochs", type=int, default=128)
p.add_argument('--midi', help="MIDI file to input", default="original_metheny.mid")
p.add_argument('--start', help="Start offset", type=float, default=478)
p.add_argument('--end', help="End offset", type=float, default=548)
p.add_argument('--bpm', type=int, default=130)
p.add_argument('--splice', type=int, default=4)
# Track numbers
p.add_argument('--melody_t', help="Track # (indexed at 0) of the melody in MIDI file", type=int, default=5)
p.add_argument('--accomp_t', help="Accompaniment tracks i.e. 1,2,3,4,5", default="0,1,6,7")
p.add_argument('--chord_t', help="Chord track", type=int, default=0)

# Model settings
p.add_argument('--max_len', type=int, default=20)
p.add_argument('--max_tries', type=int, default=1000)
p.add_argument('--max_diversity', type=int, default=0.5)



''' If run as script, execute main '''
if __name__ == '__main__':
    args = p.parse_args()
    main(args)