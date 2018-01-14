
***

![deepjazz](https://cloud.githubusercontent.com/assets/9053987/16575656/901989da-424f-11e6-9f54-6a04199e69f5.png)

### Using Keras & Theano for deep learning driven jazz generation

This is the beginning of a project built off of Ji-Sung Kim's deepjazz (https://github.com/jisungk/deepjazz), used with prior permission

 It uses Keras & Theano, two deep learning libraries, to generate jazz music. Specifically, it builds a two-layer LSTM, learning from the given MIDI file.

### Dependencies

* [Keras](http://keras.io/#installation)
* [Theano](http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions) ("bleeding-edge" version on GitHub)
* [music21](http://web.mit.edu/music21/doc/installing/index.html)

### Instructions

Run on CPU with command:
```
usage: python generator.py [-h] [--n N] [--midi MIDI] [--start START] [--end END]
                    [--bpm BPM] [--splice SPLICE] [--melody_t MELODY_T]
                    [--accomp_t ACCOMP_T] [--chord_t CHORD_T]
                    [--max_len MAX_LEN] [--max_tries MAX_TRIES]
                    [--max_diversity MAX_DIVERSITY]

Note: all arguments default to what was used for deepjazz on Metheny's "And Then I Knew"

optional arguments:
  -h, --help            show this help message and exit
  --n N                 Number of epochs
  --midi MIDI           MIDI file to input
  --start START         Start offset (equal to measure number * beat number)
  --end END             End offset (equal to measure number * beat number)
  --bpm BPM
  --splice SPLICE
  --melody_t MELODY_T   Track # (indexed at 0) of the melody in MIDI file
  --accomp_t ACCOMP_T   Accompaniment tracks to use i.e. '1,2,3,4,5'
  --chord_t CHORD_T     Chord track # to use
  --max_len MAX_LEN
  --max_tries MAX_TRIES
  --max_diversity MAX_DIVERSITY
```

Run on GPU with command:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generator.py [arguments]
```

Note: running Keras/Theano on GPU is formally supported for only NVIDIA cards (CUDA backend).

Note: `preprocess.py` must be modified to work with other MIDI files (the relevant "melody" MIDI part needs to be selected). The ability to handle this natively is a planned feature.

### Known Issues
* ValueError: sum(pvals[:-1]) > 1.0
    * I'll have to look into this more; it seems like it can be temporarily solved by using a larger sample or a smaller number of epochs
* It needs a non-rest note to be played at least every measure in both the chord and melody, or else there will be an uneven number of melody and chord measures
    * Temporary fix is selecting offsets that don't have long breaks between notes or setting splice to a higher number
* After using more complex riffs for training, it plays notes that don't belong in a chord

### Citations

This project develops a lot of preprocessing code (with permission) from Evan Chow's [jazzml](https://github.com/evancchow/jazzml). Thank you [Evan](https://www.linkedin.com/in/evancchow)! Public examples from the [Keras documentation](https://github.com/fchollet/keras) were also referenced.

### Code License, Media Copyright

Code is licensed under the Apache License 2.0  
Images and other media are copyrighted (Ji-Sung Kim)
