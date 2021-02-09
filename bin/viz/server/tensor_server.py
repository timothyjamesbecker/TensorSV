#!/usr/env/bin/python3
import os
import glob
import json
import argparse
import socket
from h5py import File
from aiohttp import web
import numpy as np

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def read_json(path):
    data = {}
    with open(path,'r') as f:
        data = json.load(f)
    return data

#if the tranform history has mean, std or min,max range entries
def parse_trans_hist(trans_hist,ftrs_idx,ms=4,dtype=np.float16):
    H,C = {},{}
    if len(trans_hist)%ms==0: #has to be a multiple of m
        for i in range(len(trans_hist)):
            fs,ts = trans_hist[i][0:2]
            if fs in C:
                if ts in C[fs]: C[fs][ts] += 1
                else:           C[fs][ts]  = 0
            else:               C[fs]  = {ts:0}
            if fs in H:
                if C[fs][ts] in H[fs]: H[fs][C[fs][ts]][ts] = dtype.type(trans_hist[i][2])
                else:                  H[fs][C[fs][ts]] = {ts:dtype.type(trans_hist[i][2])}
            else:                      H[fs] = {C[fs][ts]:{ts:dtype.type(trans_hist[i][2])}}
    F = {}
    for fs in H: F[ftrs_idx[fs]] = H[fs]
    return F

#given all.label.hdf5 tensor
def generate_ranges(label_path,out_dir):
    R = {}
    f = File(label_path,'r')
    samples  = sorted(f.keys())
    labels   = sorted(set([l for s in samples for l in f[s]]))
    if len(samples)>0 and len(labels)>0:
        dtype = f[samples[0]][labels[0]].dtype
        fs,ms = f[samples[0]][labels[0]].shape[2:]
        fo = f[samples[0]][labels[0]].attrs['feature_order']
        ftrs_idx = {fo[i]:i for i in range(len(fo))}
        for i in range(fs):
            R[i] = {}
            for j in range(ms):
                rng = [np.finfo(dtype).max,np.finfo(dtype).min]
                for s in samples:
                    for l in labels:
                        if l in f[s]:
                            l_min,l_max = np.min(f[s][l][:,:,i,j]),np.max(f[s][l][:,:,i,j])
                            if l_min<rng[0]: rng[0] = l_min #; print('new min at s=%s l=%s'%(s,l))
                            if l_max>rng[1]: rng[1] = l_max #; print('new max at s=%s l=%s'%(s,l))
                R[i][j] = [float(rng[0]),float(rng[1])]
    f.close()
    with open(out_dir+'/all.label.ranges.json','w') as f:
        json.dump(R,f)
    with open(out_dir+'/all.label.ftrs_idx.json','w') as f:
        json.dump(ftrs_idx,f)
    return R

des = """TensorSV Vizualization Server, Copyright (C) 2020-2021 Timothy James Becker"""
parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_path',type=str,help='tensor label file\t[None]')
args = parser.parse_args()
print(des)
if args.in_path is not None: in_path = args.in_path
else: raise IOError
in_dir = '/'.join(in_path.rsplit('/')[0:-1])+'/'

if not os.path.exists(in_dir+'/all.label.ranges.json'):
    print('starting building data ranges')
    ranges = generate_ranges(in_path,in_dir)
else:
    raw = read_json(in_dir+'/all.label.ranges.json')
    ranges = {}
    for fs in raw:
        ranges[int(fs)] = {}
        for ms in raw[fs]:
            ranges[int(fs)][int(ms)] = raw[fs][ms]
print('data ranges are completed')
ftrs_idx = read_json(in_dir+'/all.label.ftrs_idx.json')

f = File(in_path,'r')
sample_map = {s:sorted([l for l in f[s]]) for s in sorted(f)}
f.close()

async def sample_map_h(request):
    return web.json_response(sample_map)

#grab some instances of the sample and label
async def sample_label_h(request):
    T = {} #this will be the tensor load
    try:
        sample  = request.match_info.get('sample','Anonymous')
        label   = request.match_info.get('label', 'Anonymous')
        max_num = int(request.match_info.get('max_num', 'Anonymous'))
        if sample in sample_map and label in sample_map[sample]:
            f = File(in_path,'r')
            idx = sorted(np.random.choice(range(f[sample][label].shape[0]),min(max_num,f[sample][label].shape[0]),replace=False))
            buff = np.array(f[sample][label][idx],dtype=float)
            f.close()
            data = []
            for i in range(len(buff)):
                data += [[[[l for l in fs] for fs in row] for row in buff[i]]]
            T['idx']  = [int(x) for x in idx]
            T['data'] = data
    except Exception as E:
        print(E)
    return web.json_response(T)

#grab some instances of the label
async def label_h(request):
    T = {}  # this will be the tensor load
    try:
        label = request.match_info.get('label', 'Anonymous')
        for sample in sample_map:
            if label in sample_map[sample]:
                #read the tensors
                T = {}
    except Exception as E:
        print(E)
    return web.json_response(T)

async def ftrs_idx_h(request):
    return web.json_response(ftrs_idx)


#check port via sockets-------------------------------------------------
port = 6080
# while(port_in_use(port)): port += 1
# with open('../client/client.js','r') as f: client = f.read().rsplit('\n')
# client[0] = client[0].rsplit('localhost')[0]+"'localhost:%s';"%port
# with open('../client/client.js','w') as f: f.write('\n'.join(client))

app = web.Application()
app.router.add_get('/sample_map', sample_map_h)
app.router.add_get('/ftrs_idx', ftrs_idx_h)
app.router.add_get(r'/sample/{sample}/label/{label}/max_num/{max_num}', sample_label_h)
app.router.add_static('/', path='../client/', name='client')
web.run_app(app, host='127.0.0.1', port=port)