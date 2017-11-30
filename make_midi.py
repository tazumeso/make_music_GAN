#-*- codeing:utf8 -*-

import pretty_midi
import midi
import numpy as np
from make_music import load_train

def step_function(x, bias=0):
    return np.array(x > bias, dtype=int)

def x2arr(x):
    return step_function(x, bias=.5)

def checksize(arr):
    pass


def arr2midi(arr):
    checksize(arr)
    N, C, tick, note = arr.shape
    #arr = (arr / 255).astype(int)
    arr = arr.tolist()
    for n, music in enumerate(arr):
        pm = pretty_midi.PrettyMIDI(resolution=4, initial_tempo=120)
        instrument = pretty_midi.Instrument(0)
        time = 0
        ts = -1
        te = -1
        for t in range(tick - 1):
            try:
                if ts == -1:
                    note_number = music[0][t].index(1) + 40
                    ts = time
                if music[1][t][note_number - 40] == 0 or music[0][t+1][note_number - 40] != 1:
                    te = time + .125
                    note = pretty_midi.Note(velocity=100, pitch=note_number, start=ts, end=te)
                    instrument.notes.append(note)
                    ts = -1
            except:
                pass
            time += .125
        try:
            if ts == -1:
                note_number = music[0][tick].index(1) + 40
                ts = time
            te = time + .125
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=ts, end=te)
            instrument.notes.append(note)
        except:
            pass
        pm.instruments.append(instrument)
        pm.write('midi/music_%i.mid' % (n+1))


def test():
    arr = load_train("music_numpy")
    idx = np.random.randint(len(arr), size=10)
    arr = arr[idx, :, :, :]
    arr2midi(arr)


def main():
    test()

if __name__ == "__main__":
    main()
