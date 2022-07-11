import cv2
import numpy as np

from python_block_matching import BlockMatching
from python_block_matching.cost_functions import *
ras_blocks = set()

with open('frame10.txt') as f:
    for line in f:
        t = list(eval(line))
        t[0] -= t[2] // 2
        t[1] -= t[3] // 2
        ras_blocks.add(tuple(t))

block_info = {}

frame = cv2.imread('frame10.png')
for x, y, w, h in BlockMatching.get_macro_blocks(frame):
    v_cost = sad(frame[y:y + 16, x:x + 8], frame[y:y + 16, x + 8:x + 16])
    h_cost = sad(frame[y:y + 8, x:x + 16], frame[y + 8:y + 16, x:x + 16])
    average_color = np.sum(frame[y:y+16, x:x+16])
    if (x, y, 16, 16) in ras_blocks:
        block_info[(x, y)] = ('f', v_cost, h_cost, average_color)
    elif (x, y, 8, 16) in ras_blocks:
        block_info[(x, y)] = ('v', v_cost, h_cost, average_color)
    elif (x, y, 16, 8) in ras_blocks:
        block_info[(x, y)] = ('h', v_cost, h_cost, average_color)
    elif (x, y, 8, 8) in ras_blocks:
        block_info[(x, y)] = ('q', v_cost, h_cost, average_color)


no_cut_v, no_cut_h = [], []
h_cut_h, h_cut_v = [], []
v_cut_h, v_cut_v = [], []
q_cut_h, q_cut_v = [], []

for p, b in block_info.items():
    if b[0] == 'f':
        no_cut_v.append(b[1])
        no_cut_h.append(b[2])
    elif b[0] == 'h':
        h_cut_v.append(b[1])
        h_cut_h.append(b[2])
    elif b[0] == 'v':
        v_cut_v.append(b[1])
        v_cut_h.append(b[2])
    else:
        q_cut_v.append(b[1])
        q_cut_h.append(b[2])


no_cut_v, no_cut_h = sorted(list(set(no_cut_v))), sorted(list(set(no_cut_h)))
h_cut_v, h_cut_h = sorted(list(set(h_cut_v))), sorted(list(set(h_cut_h)))
v_cut_v, v_cut_h = sorted(list(set(v_cut_v))), sorted(list(set(v_cut_h)))
q_cut_v, q_cut_h = sorted(list(set(q_cut_v))), sorted(list(set(q_cut_h)))


print('full blocks vertical costs: ')
print(f'    {no_cut_v}')
print(f'    min={no_cut_v[0]}  max={no_cut_v[-1]}')
print('full blocks horizontal costs: ')
print(f'    {no_cut_h}')
print(f'    min={no_cut_h[0]}  max={no_cut_h[-1]}')
print('h blocks vertical costs: ')
print(f'    {h_cut_v}')
print(f'    min={h_cut_v[0]}  max={h_cut_v[-1]}')
print('h blocks horizontal costs: ')
print(f'    {h_cut_h}')
print(f'    min={h_cut_h[0]}  max={h_cut_h[-1]}')
print('v blocks vertical costs: ')
print(f'    {v_cut_v}')
print(f'    min={v_cut_v[0]}  max={v_cut_v[-1]}')
print('v blocks horizontal costs: ')
print(f'    {v_cut_h}')
print(f'    min={v_cut_h[0]}  max={v_cut_h[-1]}')
print('q blocks vertical costs: ')
print(f'    {q_cut_v}')
print(f'    min={q_cut_v[0]}  max={q_cut_v[-1]}')
print('q blocks horizontal costs: ')
print(f'    {q_cut_h}')
print(f'    min={q_cut_h[0]}  max={q_cut_h[-1]}')

zeros = []
f, h, v, q = [], [], [], []
for p, inf in block_info.items():
    if inf[0] == 'f':
        f.append(inf[1:])
    elif inf[0] == 'h':
        h.append(inf[1:])
    elif inf[0] == 'v':
        v.append(inf[1:])
    else:
        q.append(inf[1:])
    if (inf[1], inf[2]) == (0, 0):
        zeros.append(inf[0])


for i in f:
    ver, hor, ave_c = i
    print(f'f   {ver=}   {hor=}   {ave_c=}')
for i in h:
    ver, hor, ave_c = i
    print(f'h   {ver=}   {hor=}   {ave_c=}')
for i in v:
    ver, hor, ave_c = i
    print(f'v   {ver=}   {hor=}   {ave_c=}')
for i in q:
    ver, hor, ave_c = i
    print(f'q   {ver=}   {hor=}   {ave_c=}')


print(f'cuts where both v and h are {zeros=}')
