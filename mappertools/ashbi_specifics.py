"""
ASBHi-project related functions.

Top secret?
"""

import numpy as np

ashbi_colors = {"cyan":(0,255,255),
                "gold":(255,215,0),
                "tomato":(255,99,71),
                "red":(255,0,0),
                "deepskyblue":(0,191,255),
                "seagreen":(46,139,87),
                "yellowgreen":(154,205,50),
                "lightgreen":(150, 255, 150),
                #"green":(0,50,0),
                "green":(0,128,0),
                "orange":(255,165,0),
                "royalblue":(65,105,225),
                "navy":(0,0,128),
                "orange":(255,165,0),
                "brown":(165,42,42),
                "magenta":(255,0,255),
                #"gray5": (1,1,1),
                "gray5": (13,13,13),
                #"gray15": (25, 25, 25),
                "gray15": (38, 38, 38),
                #"gray50":(75,75,75),
                "gray50":(128, 128, 128),
                #"gray80":(190,190,190),
                "gray80":(204, 204, 204),
                #"slategray1":(0,0,50),
                "slategray1":(112, 128, 144),
                #"lightgray":(50,50,50),
                "lightgray":(211, 211, 211)}

def ashbi_color_averager(list_of_colors):
    list_of_triples = [ashbi_colors[clr] for clr in list_of_colors]
    ans = np.mean(np.array(list_of_triples), axis=0)
    return "rgb({:d},{:d},{:d})".format(int(ans[0]),int(ans[1]),int(ans[2]))
