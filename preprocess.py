'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Parse, cleanup and process data.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, izip_longest
from grammar import *

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn, args):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    # Get melody part, compress into single voice.
    melody_track = midi_data[args.melody_t]     # For Metheny piece, Melody is Part #5.

    """
    Some tracks (I think only polyphonic) are composed of several voices (composed of notes/rests/chords),
    others are just composed of notes/rests/chords.
    For ones with multiple voices, separate by each voice and compress into single voice
    """
    def process_track_to_stream(track):
        track_voices = track.getElementsByClass(stream.Voice)
        if len(track_voices) == 0:
            track_stream = track
        elif len(track_voices) == 1:
            track_stream = track_voices[0]
        elif len(track_voices) > 1:
            for voice in track_voices[1:]:
                for e in voice:
                    track_voices[0].insert(e.offset, e)
            track_stream = track_voices[0]
        return track_stream

    melody_stream = process_track_to_stream(melody_track)


    for i in melody_stream:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_stream.insert(0, instrument.ElectricGuitar())
    melody_stream.insert(0, key.KeySignature(1))

    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    part_indices = [int(x) for x in args.accomp_t.split(',')]

    comp_stream = stream.Voice()

    comp_stream.append([j.flat for i, j in enumerate(midi_data)
        if i in part_indices])

    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Voice()
    # Add chord data to stream
    full_stream.append(process_track_to_stream(midi_data[args.chord_t]))
    for i in xrange(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_stream)



    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)

    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        curr_part.append(part.getElementsByClass(instrument.Instrument))
        #HERE commented out below
        curr_part.append(part.getElementsByClass(tempo.MetronomeMark))
        curr_part.append(part.getElementsByClass(key.KeySignature))
        curr_part.append(part.getElementsByClass(meter.TimeSignature))
        curr_part.append(part.getElementsByOffset(args.start, args.end))
        cp = curr_part.flat
        solo_stream.insert(cp)


    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    offset_tuples = [(int(n.offset / args.splice), n) for n in melody_stream]
    measure_num = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offset_tuples, lambda x: x[0]):
        measures[measure_num] = [n[1] for n in group]
        measure_num += 1


    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chord_stream = solo_stream[0]
    chord_stream.removeByClass(note.Rest)
    chord_stream.removeByClass(note.Note)
    offset_tuples_chords = [(int(n.offset / args.splice), n) for n in chord_stream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    chord_measure_num = 0

    for key_x, group in groupby(offset_tuples_chords, lambda x: x[0]):
        chords[chord_measure_num] = [n[1] for n in group]
        chord_measure_num += 1


    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure.
    print(str(len(chords)) + "chords, measures: " + str(len(measures)))

    if len(chords) != len(measures):
        if len(chords) == len(measures) + 1:
            'WARNING: DELETED CHORD AT THE END'
            del chords[len(chords) - 1]
        if len(measures) == len(chords) + 1:
            'WARNING: DELETED MEASURE AT THE END'
            del measures[len(chords) - 1]
    assert len(chords) == len(measures)

    return measures, chords

''' Helper function to get the grammatical data from given musical data. '''
def __get_abstract_grammars(measures, chords):
    # extract grammars
    abstract_grammars = []
    for ix in xrange(1, len(measures) - 1):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        print("IX OF MEASURE" + str(ix))
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)

    return abstract_grammars

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn, args):
    measures, chords = __parse_midi(data_fn, args)
    abstract_grammars = __get_abstract_grammars(measures, chords)

    return chords, abstract_grammars

''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return corpus, values, val_indices, indices_val