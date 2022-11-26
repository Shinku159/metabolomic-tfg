from winsound import Beep
notes = {'C': 1635,
         'D': 1835,
         'E': 2060,
         'S': 1945,
         'F': 2183,
         'G': 2450,
         'A': 2750,
         'B': 3087,
         ' ': 37}
melodie = 'CDEFG'
for note in melodie:
    Beep(notes[note], 500)