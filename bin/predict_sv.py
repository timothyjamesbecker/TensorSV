#!/usr/env/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore',category=Warning)
import argparse
import time
import json
import glob
import gzip
import pickle
import itertools as it
import numpy as np
import tensorflow as tf
from tensorflow import keras
from h5py import File
import h5py
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e: print(e)
import core

def w_harm_mean(p_x):
    p = 0.0
    for i in range(len(p_x)):
        if p_x[i]>0.0: p += 1.0/p_x[i]
    if p>0.0: p = (1.0*len(p_x))/p
    return p

#p[0]=ordinal class, p[1]=prob
def class_combine(Q,ws=[1,1,1,1,1]):
    n,d,x = 0.0,0.0,Q[0][0]
    for i in range(len(Q)):
        n += Q[i][0]*Q[i][1]*ws[i]
        d += Q[i][1]*ws[i]
    if d>0.0: x = int(round(n/d))
    return x

#p[0]=pos,p[1]=prob
def pos_combine(P,ws=[1,1,1,1],w=100):
    l_pos = int(round((w*P[0][0]*P[0][1]*ws[0]+w*min(P[0][0],P[2][0])*P[2][1]*ws[2])/(P[0][1]*ws[0]+P[2][1]*ws[2])))
    r_pos = int(round((w*P[1][0]*P[1][1]*ws[1]+w*max(P[1][0],P[3][0])*P[3][1]*ws[3])/(P[1][1]*ws[1]+P[3][1]*ws[3])))
    return [l_pos,r_pos]

def overlap(c1,c2,start=1,end=2):
    if c1[end]>=c1[start] and c2[end]>=c2[start]:
        l = (c1[end]-c1[start]+1)+(c2[end]-c2[start]+1)             #total lengths
        u = min(l,max(c1[end],c2[end])-min(c1[start],c2[start])+1)  #total union area
        i = 1.0*abs(c1[start]-c2[start])+abs(c1[end]-c2[end])
        x = max(0.0,u-i)/u
    else: x = 0.0
    return x

def load_seq(tensor_in,sm,rg,seq):
    f       = File(tensor_in,'r')
    data    = np.zeros(f[sm][rg][seq].shape,dtype=f[sm][rg][seq].dtype)
    data[:] = f[sm][rg][seq][:]
    f.close()
    return data

#could make more general------------------
def mask_predictions(pred,mask,not_pos=[0]):
    for x in mask:
        pred[x[0]:x[1]] = 0.0                         # zero out the predictions on the mask
        for pos in not_pos: pred[x[0]:x[1],pos] = 1.0 #max out the NOT label for masks
    return pred

def predict_and_pack(data,models,mask):
    P,max_pred_len,classes,dtype = [],0,models[list(models.keys())[0]].outputs[0].shape[1],models[list(models.keys())[0]].dtype
    for m in models:
        P += [mask_predictions(scanning_predictor(data=data,model=models[m]),mask)]
        if P[-1].shape[0]>max_pred_len: max_pred_len = P[-1].shape[0]
    pack = np.zeros((max_pred_len,classes*len(P)),dtype=P[-1].dtype)
    for i in range(len(P)):
        pack[0:P[i].shape[0],i*classes:(i+1)*classes] = P[i][:]
        pack[P[i].shape[0]:max_pred_len,i*classes] = 1.0
    return pack

def pred_from_vca(data,models,vca,hs={1:0.5,2:1.0},w=100,flank=0.5,max_w=100,verbose=False):
    C,S,sv_idx = [],[],1
    lr_mod = keras.models.load_model(models['LR'])  #scanning model
    frames = int(lr_mod.input.shape[1]//5)
    s_i = [[0*frames,1*frames],[1*frames,2*frames],[2*frames,3*frames],[3*frames,4*frames],[4*frames,5*frames]]
    lr_n = int(lr_mod.input.shape[1])
    for vc in vca:
        if verbose: print('starting vc=%s'%vc)
        start,end,sv_len = vc[1],vc[2],vc[3]
        l,r,span = int(round(start/w)),int(round(end/w)),int(round(sv_len)/w)
        max_min_flank = min(max(5,int(round(span*flank))),max_w)
        l_rng,r_rng = [max(0,l-max_min_flank),l+max_min_flank//2],[max(0,r-max_min_flank//2),r+max_min_flank] #flanking search ranges
        x_len,y_len,xy_len = (l_rng[1]-l_rng[0]+1),(r_rng[1]-r_rng[0]+1),(r_rng[1]-l_rng[0]+1)
        #scan and search tensor grandients-------------------------------------------------------------------------
        local = np.zeros((x_len*y_len,lr_n,data.shape[1],data.shape[2]),dtype=data.dtype)
        for x in range(x_len):
            l = l_rng[0]+x
            for y in range(y_len):
                r     = r_rng[0]+y
                if l<=r:
                    xy    = x*y_len+y
                    span  = int(round((r-l)/2+1e-9))
                    a,m,b = max(0,l-span),l+span,min(r+span,data.shape[0]-5*frames)
                    local[xy,s_i[0][0]:s_i[0][1],:,:] = data[a:a+frames,:,:]
                    local[xy,s_i[1][0]:s_i[1][1],:,:] = data[l:l+frames,:,:]
                    local[xy,s_i[2][0]:s_i[2][1],:,:] = data[m:m+frames,:,:]
                    local[xy,s_i[3][0]:s_i[3][1],:,:] = data[r:r+frames,:,:]
                    local[xy,s_i[4][0]:s_i[4][1],:,:] = data[b:b+frames,:,:]
        sv_diff = np.argmin([abs(sv_len-w*frames),abs(sv_len-2*w*frames),abs(sv_len-3*w*frames)])
        lr_pred = lr_mod.predict(local)
        #------------------------------------------------------------------------------------------------------------
        lr_max = [0.0,0,0]
        for c in range(1,lr_pred.shape[1],1):
            idx = np.argmax(lr_pred[:,c])
            if lr_pred[idx,c]>lr_max[0]: lr_max = [lr_pred[idx,c],idx,c]
        lr_pos = [l_rng[0]+lr_max[1]//y_len, r_rng[0]+lr_max[1]%y_len]
        if (lr_pos[1]-lr_pos[0]+1)*w>0:
            C += [[vc[0],lr_pos[0]*w,lr_pos[1]*w,(lr_pos[1]-lr_pos[0]+1)*w,hs[lr_max[2]],lr_max[0],2-lr_max[2],vc[-1]]]
    return C,S

#large scale scanning predictor for large chrom regions and GPU acceleration
def scanning_predictor(data,model,batch_size=4096):
    ws      = data.shape
    dt      = data.dtype
    n_d     = int(model.input.shape[1])
    classes = int(model.output.shape[1])
    if batch_size>ws[0]: batch_size = ws[0]
    buff = np.zeros((batch_size,n_d,ws[1],ws[2]),dtype=dt)
    pred = np.zeros((ws[0]-n_d,classes),dtype=np.float32)
    bs   = [b for b in range(0,ws[0]-n_d,batch_size)]
    if (ws[0]-n_d)%batch_size>0: bs += [bs[-1]+(ws[0]-n_d)%batch_size]
    start = time.time()
    for j in range(len(bs)-1):
        for i in range(bs[j+1]-bs[j]):
            buff[i] = data[bs[j]+i:bs[j]+n_d+i]
        pred[bs[j]:bs[j+1],:] = model.predict(buff[:bs[j+1]-bs[j],:])
    stop = time.time()
    return pred

#get prediction arround start and end coordinates
def pred_true(pred,T,w,offset=5,flank=5):
    a = (T[1]//w)+offset
    b = (T[2]//w)+offset
    if a<=b: b = a+1
    return pred[max(0,a-flank):min(b+flank,len(pred)),:]

#given a list of indecies, form ranges
def idx_to_ranges(idx):
    R = []
    if len(idx)>0:
        R += [[idx[0]]]
        for i in range(1,len(idx),1):
            if idx[i]-1<=R[-1][-1]: R[-1] += [idx[i]]
            else:                   R     += [[idx[i]]]
        for i in range(len(R)): R[i] = [R[i][0],R[i][-1]]
    return R

def get_ftr_idx(data):
    return {data.attrs['feature_order'][i]: i for i in range(len(data.attrs['feature_order']))}

def get_trans_hist(data,ms):
    trans_hist = data.attrs['trans_hist']
    th = {}
    for i in range(0,len(trans_hist),ms*2):
        ft = trans_hist[i][0]
        th[ft] = {0:[float(trans_hist[i+0][-1]),float(trans_hist[i+1][-1])],
                  1:[float(trans_hist[i+2][-1]),float(trans_hist[i+3][-1])],
                  2:[float(trans_hist[i+4][-1]),float(trans_hist[i+5][-1])],
                  3:[float(trans_hist[i+6][-1]),float(trans_hist[i+7][-1])]}
    return th

#get the ranges of the breaks for left
def get_breaks(pred,w,min_p=0.9,classes=3,method='full'):
    brks = []
    for i in range(len(pred)):
        m = np.argmax(pred[i,1:classes])+1
        if pred[i,m]>min_p: brks += [i]
    B,P = [],[]
    if len(brks)>0:
        B += [[brks[0]]]
        for i in range(1,len(brks),1):
            if brks[i]-1<=B[-1][-1]: B[-1] += [brks[i]]
            else:                    B     += [[brks[i]]]
        for i in range(len(B)): B[i] = [B[i][0],B[i][-1]]
        P = []
        if method=='max':
            for j in range(len(B)):
                maxs = [min_p,0]
                for c in range(1,classes):
                    i = np.argmax(pred[B[j][0]:B[j][1]+1,c])
                    m = pred[B[j][0]+i,c]
                    if pred[B[j][0]+i,c]>maxs[0]: maxs = [m,B[j][0]+i]
                P += [[maxs[1]*w,maxs[0]]]
            P = sorted(P,key=lambda x: x[1])[::-1]
        elif method=='full':
            for j in range(len(B)):
                ps = np.mean(np.sum(pred[B[j][0]:B[j][1]+1,1:classes],axis=1))
                P += [[B[j][0]*w,B[j][1]*w,ps]]
            P = sorted(P,key=lambda x: x[2])[::-1]
    return P

#left to left lists are within flank?
def test_brk(L,P,flank=300):
    n,m = len(L),len(P)
    a,b = set([]),set([])
    for i,j in it.product(range(n),range(m)):
        if abs(L[i]-P[j])<=flank:
            a.add(i)
            b.add(j)
    prec,rec = (len(a)/n if n>0 else 0.0),(len(b)/m if m>0 else 0.0)
    f1 = (2.0*(prec*rec)/(prec+rec) if prec+rec>0.0 else 0.0)
    return {'prec':prec,'rec':rec,'f1':f1,'n':n,'m':m}

def test_rngs(A,B,r=0.5):
    n,m = len(A),len(B)
    a,b = set([]),set([])
    for i,j in it.product(range(n),range(m)):
        if overlap(A[i],B[j],start=0,end=1)>=r:
            a.add(i)
            b.add(j)
    prec,rec = (len(a)/n if n>0 else 0.0),(len(b)/m if m>0 else 0.0)
    f1 = (2.0*(prec*rec)/(prec+rec) if prec+rec>0.0 else 0.0)
    return {'prec':prec,'rec':rec,'f1':f1,'n':n,'m':m}

def opt_filter(T,data,ftr_idx,trk,w,values,comps=['<='],flank=500,opt='f1'):
    V = {c:[{'prec':0.0,'rec':0.0,'f1':0.0,'n':0.0,'m':0.0},0.0,[]] for c in comps}
    for v in values:
        for c in comps:
            if c=='<':  rngs  = idx_to_ranges(np.where(data[:,ftr_idx[trk],0]<v)[0])
            if c=='<=': rngs  = idx_to_ranges(np.where(data[:,ftr_idx[trk],0]<=v)[0])
            if c=='>':  rngs  = idx_to_ranges(np.where(data[:,ftr_idx[trk],0]>v)[0])
            if c=='>=': rngs  = idx_to_ranges(np.where(data[:,ftr_idx[trk],0]>=v)[0])
            score = test_brk(T,[x[0]*w for x in rngs],flank=flank)
            if score[opt]>V[c][0][opt]: V[c] = [score,v,rngs]
    return V

#undo the transformation history
def reverse_transform(data,th,ftr_idx,ms):
    D = np.zeros(data.shape,np.float32)
    for i in ftr_idx:
        for j in range(ms):
            mean = th[i][j][0]
            std  = th[i][j][1]
            D[:,ftr_idx[i],j] = (data[:,ftr_idx[i],j]*std)+mean
    return D

def scan_breaks(pred,m_labels,classes):
    p_len,bb,mns = len(pred),[],{i:m_labels[i] for i in range(len(m_labels))}
    brks,B = [],[]
    for b in range(pred.shape[1]//classes):
        brks += [[np.argmax(x) for x in pred[:,b*classes:(b+1)*classes]]]
    for b in range(len(brks)): bb += [{i:[] for i in range(1,classes,1)}]
    for b in range(len(brks)):
        for i in range(len(brks[b])):
            for c in bb[b]:
                if brks[b][i]==c: bb[b][c] += [i]
    for b in range(len(bb)):
        B += [{}]
        for c in bb[b]:
            if len(bb[b][c])>0: B[b][c] = [[bb[b][c][0]]]
    for b in range(len(B)):
        for c in B[b]:
            for i in range(1,len(bb[b][c])):
                if bb[b][c][i]-1<=B[b][c][-1][-1]: B[b][c][-1] += [bb[b][c][i]]
                else:                              B[b][c]     += [[bb[b][c][i]]]
            for i in range(len(B[b][c])):          B[b][c][i]   = [B[b][c][i][0],B[b][c][i][-1]]
            B[b][c] = [[B[b][c][i][0],B[b][c][i][1],mns[b],c] for i in range(len(B[b][c]))]
    return B

def cluster_knn_pairs(B,pred,mask,k=3,order=None,trim=0.5,min_p=0.9,use_peak=False,verbose=False):
    AB,CD,classes = {},{},len(B[0])+1
    m_map = {}
    for b in range(len(B)):
        if len(B[b].keys())>0:
            m = B[b][list(B[b].keys())[0]][0][2]
            m_map[m] = b
    m_map_idx = {m_map[x]:x for x in m_map}
    if verbose: print('--- starting clustering with %s models and %s classes with k=%s ---'%(len(m_map),classes,k))

    for a,b in it.combinations(range(len(B)),2):
        (a,b) = sorted([a,b])
        AB[(a,b)] = {}
        for c in B[b]:
            if c in B[a]:
                AB[(a,b)][c] = sorted(B[b][c]+B[a][c],key=lambda x: x[0])
    for a,b in AB:
        CD[(a,b)] = {}
        for c in AB[(a,b)]:
            CD[(a,b)][c] = {}
            for i in range(len(AB[(a,b)][c])):
                pk = AB[(a,b)][c][i]
                CD[(a,b)][c][tuple(pk)] = {-1:[],1:[]} #-1 is <= pk, 1 is >= pk
                for j in range(i-1,0,-1): #go left
                    if AB[(a,b)][c][j][2]!=pk[2]:
                        CD[(a,b)][c][tuple(pk)][-1] += [AB[(a,b)][c][j]]
                    if len(CD[(a,b)][c][tuple(pk)][-1])>=k:
                        break
                for j in range(i+1,len(AB[(a,b)][c]),1): #go right
                    if AB[(a,b)][c][j][2]!=pk[2]:
                        CD[(a,b)][c][tuple(pk)][1] += [AB[(a,b)][c][j]]
                    if len(CD[(a,b)][c][tuple(pk)][1])>=k:
                        break
    AB,CD=CD,{} #AB[(L,R)][0.5][cluster1]:knn
    for a,b in AB:
        CD[(a,b)] = {}
        for c in AB[(a,b)]:
            CD[(a,b)][c] = {}
            for q in AB[(a,b)][c]:
                q_p = np.max(pred[q[0]:q[1]+1,m_map[q[2]]*classes+c])
                for s in AB[(a,b)][c][q]:
                    for l in AB[(a,b)][c][q][s]:
                        if q[0]<=l[0]:
                            v = tuple([q,tuple(l)])
                            if v not in CD[(a,b)][c]:
                                l_p = np.max(pred[l[0]:l[1]+1,m_map[l[2]]*classes+c])
                                CD[(a,b)][c][v] = [q_p,l_p]
                        else:
                            v = tuple([tuple(l),q])
                            if v not in CD[(a,b)][c]:
                                l_p = np.max(pred[l[0]:l[1]+1,m_map[l[2]]*classes+c])
                                CD[(a,b)][c][v] = [l_p,q_p]
    AB,CD=CD,{}

    if verbose:
        print('::: starting filtering of low p-value breakpoints :::')
        print('------------------------------------------------------------------------------')
    for a,b in AB:
        CD[(a,b)] = {}
        for c in AB[(a,b)]:
            CD[(a,b)][c] = []
            for q in AB[(a,b)][c]:
                w_p = w_harm_mean(AB[(a,b)][c][q])
                if w_p>=min_p: CD[(a,b)][c] += [[q[0],q[1],c,AB[(a,b)][c][q],[q[0][2],q[1][2]]]]
            CD[(a,b)][c] = sorted(CD[(a,b)][c],key=lambda x: min(x[0][0],x[1][0]))
            if verbose: print('removed %s low p-value brkpts out of %s for model-pair=%s class=%s'%(len(AB[(a,b)][c])-len(CD[(a,b)][c]),len(AB[(a,b)][c]),(a,b),c))
    AB,CD=CD,{}

    #remove non-ordered and then reorder pairs according to order argument
    if order is not None:
        for a,b in AB:
            CD[(a,b)] = {}
            for c in AB[(a,b)]:
                CD[(a,b)][c],j = [],0
                for i in range(len(AB[(a,b)][c])):
                    x,y = order[AB[(a,b)][c][i][0][2]],order[AB[(a,b)][c][i][1][2]]
                    if x<=y: CD[(a,b)][c] += [AB[(a,b)][c][i]]
                    else:    j += 1
                if verbose:print('removed %s improperly ordered clusters for model-pair=%s class=%s'%(j,(a,b),c))
        AB,CD=CD,{}
        for a,b in AB:
            if a in m_map_idx and b in m_map_idx:
                a_m,b_m = m_map_idx[a],m_map_idx[b]
                cd = tuple(sorted([order[a_m],order[b_m]]))
                CD[cd] = {}
                for c in AB[(a,b)]:
                    CD[cd][c] = []
                    for i in range(len(AB[(a,b)][c])):
                        CD[cd][c] += [AB[(a,b)][c][i]]
        if verbose:
            print('reordered pairs according to order=%s'%order)
            print('------------------------------------------------------------------------------')
        AB,CD=CD,{}
    for a,b in AB:
        for c in AB[(a,b)]:
            AB[(a,b)][c] = sorted(AB[(a,b)][c],key=lambda x: x[0][0])
    #remove intersections:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    if trim>0.0:
        t_start = time.time()
        if verbose: print('::: calculating intersecting high p-valued model-pair clusters :::')
        I = {}
        for a,b in AB:
            I[(a,b)] = {}
            for c in AB[(a,b)]:
                I[(a,b)][c] = {}
                for i in range(len(AB[(a,b)][c])):
                    q  = AB[(a,b)][c][i]
                    if use_peak:
                        r1 = q[0][0]+np.argmax(pred[q[0][0]:q[0][1]+1,m_map[q[0][2]]*classes+c])
                        r2 = q[1][0]+np.argmax(pred[q[1][0]:q[1][1]+1,m_map[q[1][2]]*classes+c])
                    else: r1,r2 = q[0][0],q[1][1]
                    c1 = [r1,r2,w_harm_mean(q[3])]
                    for j in range(i+1,len(AB[(a,b)][c]),1): #scan right only-----------------------------
                        p  = AB[(a,b)][c][j]
                        if use_peak:
                            t1 = p[0][0]+np.argmax(pred[p[0][0]:p[0][1]+1,m_map[p[0][2]]*classes+c])
                            t2 = p[1][0]+np.argmax(pred[p[1][0]:p[1][1]+1,m_map[p[1][2]]*classes+c])
                        else: t1,t2 = p[0][0],p[1][1]
                        c2 = [t1,t2,w_harm_mean(p[3])]
                        if c2[0]<=c1[1]: #boundry check-----------------------
                            if overlap(c1,c2,start=0,end=1)>trim:
                                if c1[2]>c2[2]:
                                    if i in I[(a,b)][c]: I[(a,b)][c][i] += [j]
                                    else:                I[(a,b)][c][i]  = [j]
                                else:
                                    if j in I[(a,b)][c]: I[(a,b)][c][j] += [i]
                                    else:                I[(a,b)][c][j]  = [i]
                        else: break
        R = {}
        for a,b in I:
            R[(a,b)] = {}
            for c in I[(a,b)]:
                R[(a,b)][c] = set([])
                ks = sorted(list(I[(a,b)][c].keys()))
                for i in ks:
                    if i in I[(a,b)][c]:
                        for j in I[(a,b)][c][i]:
                            R[(a, b)][c].add(j)
                            if j in I[(a,b)][c]: I[(a,b)][c].pop(j)
        for a,b in AB:
            CD[(a,b)] = {}
            for c in AB[(a,b)]:
                CD[(a,b)][c],j = [],0
                for i in range(len(AB[(a,b)][c])):
                    if i not in R[(a,b)][c]:
                        CD[(a,b)][c] += [AB[(a,b)][c][i]]
                    else: j += 1
                if verbose: print('removed %s intersecting lower p-valued clusters from model-pair=%s class=%s'%(j,(a,b),c))
        AB,CD=CD,{}
        t_stop = time.time()
        if verbose:
            print('------------------------------------------------------------------------------')
            for a,b in AB:
                for c in AB[(a,b)]:
                    print('%s clusters left for p-value>%s model-pair=%s k=%s class=%s'%(len(AB[(a,b)][c]),min_p,(a,b),k,c))
            print('intersection time was %s sec'%round(t_stop-t_start,2))
    return AB

#cord sorted cluster calls
def cluster_calls(C,seq,pred,classes,m_map={'L':0,'R':1},hs={1:0.5,2:1.0},w=25,n=11):
    V,x = [],0
    for c in C[(m_map['L'],m_map['R'])]:
        for i in range(len(C[(m_map['L'],m_map['R'])][c])):
            vc = C[(m_map['L'],m_map['R'])][c][i]
            r1 = vc[0][0]+np.argmax(pred[vc[0][0]:vc[0][1]+1,m_map[vc[0][2]]*classes+c])
            r2 = vc[1][0]+np.argmax(pred[vc[1][0]:vc[1][1]+1,m_map[vc[1][2]]*classes+c])
            x += 1
            V += [[seq,r1*w+n*w,r2*w+n*w,(r2*w-r1*w+1*w),hs[vc[2]],w_harm_mean(vc[3]),2-vc[2],'idx_%s'%x]]
    V = sorted(V,key=lambda x: x[1])
    return V

def lr_breaks_to_calls(B,seq,pred,min_p=0.9,w=25,n=11):
    V,x = [],0
    for k in B:
        for i in range(len(B[k])):
            vc = B[k][i]
            max_i  = np.argmax(pred[vc[0]:vc[1]+1,vc[3]])
            max_l  = pred[vc[0]+max_i,vc[3]]
            if max_l>=min_p:
                r      = vc[0]+max_i
                x += 1
                V += [[seq,vc[0]*w+n*w,r*w+5*n*w,(r*w+5*n*w)-(vc[0]*w+n*w),hs[vc[3]],pred[vc[0]+max_i,vc[3]],vc[3],'idx_%s'%x]]
    V = sorted(V,key=lambda x: x[1])
    return V

#merge calls within C that have r overlap using either
#min: the minimum number of calls to elliminate conflict
#size: keep larger calls to elliminate conflict
#like: use liklihood scores to elliminate
#multi: use multiple weights from min,size,liklihood
def merge_calls(C,r=0.5,strategy='min',target=None,verbose=False):
    S,M = {},{}
    D,C2 = {},[] #remove exact duplicates: seq,start,end,geno
    for i in range(len(C)):
        k = (C[i][0],C[i][1],C[i][2],C[i][3])
        if k in D: D[k] += [i]
        else:      D[k]  = [i]
    for d in D: C2 += [C[D[d][0]]]
    C2 = sorted(C2,key=lambda x: (x[0].zfill(255),x[1])) #sort by seq,start
    for i in range(len(C2)):
        if C2[i][0] in S: S[C2[i][0]] += [C2[i]]
        else:             S[C2[i][0]]  = [C2[i]]
    for seq in S:
        #(1) find overlap conflicts where overlap>=r
        A,k = {i:set([]) for i in range(len(S[seq]))},1 #number of conflicts
        for i in range(len(S[seq])):
            for j in range(k,len(S[seq]),1):
                if r<=overlap(S[seq][i],S[seq][j]):
                    A[i].add(j)
                    A[j].add(i)
            k += 1
        conflicts = {}
        for a in A:
            if len(A[a])>0: conflicts[a] = len(A[a])
        if len(conflicts)>0 and verbose: print('seq=%s:\tmerging %s call conflicts out of %s calls'%(seq,len(conflicts),len(S[seq])))
        elif verbose:                    print('seq=%s: no conflict calls to merge'%seq)

        #(2) remove conflicts based on #of conflics and the likelihoods
        if strategy=='min':
            while len(conflicts)>0:
                #select conflict idx----------------------------------
                c = sorted(conflicts)[0]
                max_c = [conflicts[c],c]
                for c in conflicts:
                    if conflicts[c]>max_c[0]: max_c = [conflicts[c],c]
                idx = max_c[1]

                #update conflicts----------------------------
                A.pop(idx)
                for a in A: A[a] = A[a].difference(set([idx]))
                conflicts = {}
                for a in A:
                    if len(A[a])>0: conflicts[a] = len(A[a])
        elif strategy=='size':
            while len(conflicts)>0:
                #select conflict idx----------------------------------
                c = sorted(conflicts)[0]
                b = sorted(A[c])[0]
                max_c = [S[seq][b][3],b]
                for b in A[c]:
                    if S[seq][b][3]>S[seq][c][3]: max_c = [S[seq][c][3],c]
                idx = max_c[1]

                #update conflicts----------------------------
                A.pop(idx)
                for a in A: A[a] = A[a].difference(set([idx]))
                conflicts = {}
                for a in A:
                    if len(A[a])>0: conflicts[a] = len(A[a])
        elif strategy=='like':
            while len(conflicts)>0:
                #select conflict idx----------------------------------
                #select conflict idx----------------------------------
                c = sorted(conflicts)[0]
                b = sorted(A[c])[0]
                max_c = [S[seq][b][3],b]
                for b in A[c]:
                    if S[seq][c][5]<S[seq][b][5]: max_c = [S[seq][c][3],c]
                idx = max_c[1]

                #update conflicts----------------------------
                A.pop(idx)
                for a in A: A[a] = A[a].difference(set([idx]))
                conflicts = {}
                for a in A:
                    if len(A[a])>0: conflicts[a] = len(A[a])
        elif strategy=='multi':
            while len(conflicts)>0:
                #select conflict idx----------------------------------
                c = sorted(conflicts)[0]
                max_c = [conflicts[c],c]
                for c in conflicts:
                    if conflicts[c]>max_c[0]: max_c = [conflicts[c],c]
                idx = max_c[1]

                #update conflicts----------------------------
                A.pop(idx)
                for a in A: A[a] = A[a].difference(set([idx]))
                conflicts = {}
                for a in A:
                    if len(A[a])>0: conflicts[a] = len(A[a])

        #(3) sort by likelihoods and select the top target number if not None
        S[seq] = [S[seq][idx] for idx in A]
        if type(target) is dict:
            for seq in target:
                S[seq] = sorted(S[seq],key=lambda x: x[5])[::-1][0:int(0.5+min(target[seq],len(S[seq])))]
                S[seq] = sorted(S[seq],key=lambda x: (x[0].zfill(255),x[1]))
    return S

def model_scanner(tensor_in,sm,rg,seq,mask,models,order=None,hs={1:0.5,2:1.0},k=3,min_p=0.5,trim=0.25,w=100,verbose=False):
    #-----LOAD-DATA--------------------------------------------------------------------
    start    = time.time()
    f       = File(tensor_in,'r')
    data    = np.zeros(f[sm][rg][seq].shape,dtype=f[sm][rg][seq].dtype)
    data[:] = f[sm][rg][seq][:]
    f.close()
    stop = time.time()
    print('loaded sm=%s, rg=%s, seq=%s data in %s sec'%(sm,rg,seq,round(stop-start,2)))
    #------SCAN-MODELS-----------------------------------------------------------------
    M = {}
    for m in sorted(models.keys()):
        if m.find('-LR')<0: M[m] = keras.models.load_model(models[m])
    classes = int(M[list(M.keys())[0]].output.shape[1]) #should be all the same here
    pred = predict_and_pack(data,M,[[x[0]//w,x[1]//w] for x in mask[seq]])
    stop  = time.time()
    print('scanned models=%s and applied mask regions in %s sec'%(M.keys(),round(stop-start,2)))
    #:::::::CLUSTER:::PREDICTIONS:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    start = time.time() #scan the likelihood estimates for L and R breakpoint ranges
    B = scan_breaks(pred,m_labels=list(M.keys()),classes=classes)
    C = cluster_knn_pairs(B,pred,mask,k=1,order=order,trim=0.1,min_p=0.95,verbose=verbose)
    K = cluster_calls(C,seq,pred,classes=classes,m_map=order,hs=hs,min_mean_p=0.95)
    print(sv_score(T[seq],K))
    stop = time.time() #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    print('model scanning and clustering completed in %s sec'%round(stop-start,2))
    return K

def coord_to_w(vc,w=100,f=3):
    a,b = int(round(vc[1]/w)),int(round(vc[2]/w))
    if b-a<1: b = a+1
    a,b = a-f,b+f
    return [a,b]

def viz_coord(c):
    return '%s:%s-%s'%(c[0],c[1],c[2])

#have set of coord-sorted sv_type calls C1, C2, score them
def sv_score(C1,C2,r=0.5,rnd=2):
    if type(C1) is list and type(C2) is list:
        a,h,b,k,n,m = set([]),set([]),set([]),set([]),len(C1),len(C2)
        for i,j in it.product(range(n),range(m)):
            if r<=overlap(C1[i],C2[j]):
                a.add(i)
                b.add(j)
                if C1[i][4]==C2[j][4]: h.add(j)
                else:                  k.add(j)
        prec,rec,f1 = 0.0,0.0,0.0
        if n>0:             prec = 1.0*len(a)/n
        if m>0:             rec  = 1.0*len(b)/m
        if prec+rec>0.0:    f1 = 2.0*(prec*rec)/(prec+rec)
        if len(h)+len(k)>0: hk = 1.0*len(h)/(len(h)+len(k))
        else:               hk = 0.0
        a_t,b_t,h_t,k_t,n_t,m_t = len(a),len(b),len(h),len(k),n,m
        F = core.LRF_1D(sorted([[c1[1],c1[2]] for c1 in C1]),sorted([[c2[1],c2[2]] for c2 in C2]))
        sum_I,sum_U = float(sum([x[1]-x[0]+1 for x in F[0]])),float(sum([x[1]-x[0]+1 for x in F[1]]))
        if sum_U>0.0: sim = sum_I/sum_U
        else:         sim = 0.0
    elif type(C1) is dict and type(C2) is dict:
        a_t,b_t,h_t,k_t,n_t,m_t,sum_I,sum_U = 0,0,0,0,0,0,0.0,0.0
        for seq in C1:
            a,h,b,k,n,m = set([]),set([]),set([]),set([]),len(C1[seq]),0
            if seq in C2:
                m += len(C2[seq])
                for i,j in it.product(range(n),range(m)):
                    if r<=overlap(C1[seq][i],C2[seq][j]):
                        a.add(i)
                        b.add(j)
                        if C1[seq][i][4]==C2[seq][j][4]: h.add(j)
                        else:                            k.add(j)
                F = core.LRF_1D(sorted([[c1[1],c1[2]] for c1 in C1[seq]]),sorted([[c2[1],c2[2]] for c2 in C2[seq]]))
                sum_I += float(sum([x[1]-x[0]+1 for x in F[0]]))
                sum_U += float(sum([x[1]-x[0]+1 for x in F[1]]))
            else:
                sum_I += 0.0
                sum_U += float(sum([x[2]-x[1]+1 for x in C1[seq]]))
            a_t,b_t,h_t,k_t,n_t,m_t = a_t+len(a),b_t+len(b),h_t+len(h),k_t+len(k),n_t+n,m_t+m
        prec,rec,f1,sim = 0.0,0.0,0.0,0.0
        if n_t>0:        prec = 1.0*a_t/n_t
        if m_t>0:        rec  = 1.0*b_t/m_t
        if prec+rec>0.0: f1 = 2.0*(prec*rec)/(prec+rec)
        if h_t+k_t>0:    hk = 1.0*h_t/(h_t+k_t)
        else:            hk = 0.0
        if sum_U>0.0:    sim = sum_I/sum_U
        else:            sim = 0.0
    return {'prec':round(prec,rnd),'rec':round(rec,rnd),'f1':round(f1,rnd),'j':round(sim,rnd),
            'hk':round(hk,rnd),'a':a_t,'b':b_t,'n':n_t,'m':m_t,'h':h_t,'k':k_t}

#have large calls with likelihood estimates and true
def opt_score(V,T,metric='f1',r=0.5,rnd=2):
    print('optimizing %s score...'%metric)
    start = time.time()
    rngs = np.arange(0.1,10.1,0.1)
    max_rng = [0.0,0]
    for i in range(len(rngs)):
        target = {}
        for row in V:
            if row[0] in T and not row[0] in target: target[row[0]] = int(round(rngs[i]*len(T[row[0]])))
        ts = {}
        for seq in T:
            if seq in target: ts[seq] = T[seq]
        SV = merge_calls(V,r=r,strategy='like',target=target)
        score = sv_score(ts,SV,r=r,rnd=16)[metric]
        if score>=max_rng[0]: max_rng = [score,i,target]
    stop = time.time()
    print('optimal: %s=%s at %sx%s in %s sec'%(metric,round(max_rng[0],rnd),max_rng[2],rngs[max_rng[1]],round(stop-start,2)))
    return max_rng[0]

def score_samples(sample_dir,out_dir,sv='DEL',size_bin=[0,int(1E5)],geno_bin=[0.0,0.25],target=2500):
    S = {}
    for path in sorted(glob.glob(sample_dir+'/*%s*.pickle.gz'%sv)):
        with gzip.GzipFile(path,'rb') as f:
            D = pickle.load(f)
            sm = sorted(D)[0]
            sv = sorted(D[sm])[0]
            if sm not in S:     S[sm]     = {}
            if sv not in S[sm]: S[sm][sv] = {}

            if 'ts' in D[sm][sv]['vca']:
                F = {}
                for seq in D[sm][sv]['vca']['ts']:
                    for vc in D[sm][sv]['vca']['ts'][seq]:
                        if vc[3]>size_bin[0] and vc[3]<=size_bin[1] and vc[4]>geno_bin[0] and vc[4]<=geno_bin[1]:
                            if seq in F: F[seq] += [vc]
                            else:        F[seq]  = [vc]
                D[sm][sv]['vca']['ts'] = F
                if 'tensor' in D[sm][sv]['vca']:
                    if 'LR' in D[sm][sv]['vca']['tensor']:
                        F = []
                        for vc in D[sm][sv]['vca']['tensor']['LR']:
                            if vc[3]>size_bin[0] and vc[3]<=size_bin[1] and vc[4]>geno_bin[0] and vc[4]<=geno_bin[1]:
                                F += [vc]
                        D[sm][sv]['vca']['tensor']['LR'] = F
                        V = sorted(D[sm][sv]['vca']['tensor']['LR'],key=lambda x: x[5])[::-1][0:min(len(D[sm][sv]['vca']['tensor']['LR']),target)]
                        SV  = {}
                        for v in V:
                            if v[0] in SV: SV[v[0]] += [v]
                            else:          SV[v[0]]  = [v]
                        S[sm][sv]['tensor']['LR'] = sv_score(D[sm][sv]['vca']['LR']['ts'],SV)
                    elif 'L,R' in D[sm][sv]['vca']['tensor']:
                        F = []
                        for vc in D[sm][sv]['vca']['tensor']['L,R']:
                            if vc[3]>size_bin[0] and vc[3]<=size_bin[1] and vc[4]>geno_bin[0] and vc[4]<=geno_bin[1]:
                                F += [vc]
                        D[sm][sv]['vca']['tensor']['L,R'] = F
                        V = sorted(D[sm][sv]['vca']['tensor']['L,R'],key=lambda x: x[5])[::-1][0:min(len(D[sm][sv]['vca']['tensor']['L,R']),target)]
                        SV  = {}
                        for v in V:
                            if v[0] in SV: SV[v[0]] += [v]
                            else:          SV[v[0]]  = [v]
                        S[sm][sv]['tensor']['L,R'] = sv_score(D[sm][sv]['vca']['L,R']['ts'],SV)
                    else:
                        F = []
                        for vc in D[sm][sv]['vca']['tensor']:
                            if vc[3]>size_bin[0] and vc[3]<=size_bin[1] and vc[4]>geno_bin[0] and vc[4]<=geno_bin[1]:
                                F += [vc]
                        D[sm][sv]['vca']['tensor'] = F
                        V = sorted(D[sm][sv]['vca']['tensor'],key=lambda x: x[5])[::-1][0:min(len(D[sm][sv]['vca']['tensor']),target)]
                        SV  = {}
                        for v in V:
                            if v[0] in SV: SV[v[0]] += [v]
                            else:          SV[v[0]]  = [v]
                        S[sm][sv]['tensor'] = sv_score(D[sm][sv]['vca']['ts'],SV)
                if 'comp' in D[sm][sv]['vca']:
                    F = {}
                    for seq in D[sm][sv]['vca']['comp']:
                        for vc in D[sm][sv]['vca']['comp'][seq]:
                            if vc[3]>size_bin[0] and vc[3]<=size_bin[1] and vc[4]>geno_bin[0] and vc[4]<=geno_bin[1]:
                                if seq in F: F[seq] += [vc]
                                else:        F[seq]  = [vc]
                    D[sm][sv]['vca']['comp'] = F
                    S[sm][sv]['comp'] = sv_score(D[sm][sv]['vca']['ts'],D[sm][sv]['vca']['comp'])
            else:
                print('ts entry was not found, no calls to compare for measure...')
    return S

def plot_scans(preds,rng,offs,classes): # rng is the range of plotting m = #models, c = #classes
    colors = ['purple','blue','green','yellow','orange','red','black','brown'] # up to L,M,R, models=3
    legend,x = [],0
    for i in range(preds.shape[1]//classes): #0,1,2
        for j in range(classes): #0,1,2
            if j>0:              #i=0 0*1+1=1, 0*2+2=2, 1*1+1=2
                plt.plot(preds[rng[0]+offs[i]:rng[1]+offs[i],i*classes+j],color=colors[x])
                legend += ['m=%s c=%s'%(i+1,j)]
                x += 1
    plt.legend(legend)
    plt.ylim(0.0,1.0)
    plt.show()

if __name__ == '__main__':
    des = """predict_sv: TensorSV Prediction Framework v0.1.2\nCopyright (C) 2020-2021 Timothy James Becker\n"""
    parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--base_dir',type=str,help='base directory that has normalized tensors and targets\t[None]')
    parser.add_argument('--run_dir',type=str,help='training run directory\t[None]')
    parser.add_argument('--out_dir',type=str,help='pickle/VCF outut directory\t[None]')
    parser.add_argument('--samples',type=str,help='comma seperated list of sample names to take from base_directory\t[all]')
    parser.add_argument('--sv_type',type=str,help='DEL,DUP,CNV,INV,INS\t[DEL]')
    parser.add_argument('--seqs',type=str,help='comma seperated sequences to predict on\t[all]')
    parser.add_argument('--gpu_num',type=int,help='pick one of your available logical gpus\t[None]')
    args = parser.parse_args()
    if args.base_dir is not None:
        base_dir = args.base_dir
    else: raise IOError
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        print('run directory argument is empty')
        raise IOError
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        print('out directory argument is empty')
        raise IOError
    if args.samples is not None:
        sms = args.samples.split(',')
        samples = {}
        norms = sorted(glob.glob(base_dir+'/tensors/*.norm.hdf5'))
        for norm in norms:
            sm = norm.split('/')[-1].split('.')[0]
            if sm in sms: samples[sm] = norm
    else:
        samples = {}
        norms = sorted(glob.glob(base_dir+'/tensors/*.norm.hdf5'))
        for norm in norms:
            sm = norm.split('/')[-1].split('.')[0]
            samples[sm] = norm
    if args.sv_type is not None:
        sv = args.sv_type
    else: sv = 'DEL'
    if args.seqs is not None:
        selected_seqs = args.seqs.rsplit(',')
    else:
        selected_seqs = 'all'
    if args.gpu_num is not None:        gpu_num  = args.gpu_num
    else:                               gpu_num  = 0

    model_path  = run_dir+'/test/models/all-%s-L.model.hdf5'%sv
    if os.path.exists(model_path):
        five_models = {'L': run_dir+  '/test/models/all-%s-L.model.hdf5'%sv,
                       'R': run_dir+  '/test/models/all-%s-R.model.hdf5'%sv,
                       'LR': run_dir+ '/test/models/all-%s-LR.model.hdf5'%sv}
    else:
        five_models = {'L': run_dir+  '/models/all-%s-L.model.hdf5'%sv,
                       'R': run_dir+  '/models/all-%s-R.model.hdf5'%sv,
                       'LR': run_dir+ '/models/all-%s-LR.model.hdf5'%sv}

    model_base = five_models[sorted(five_models)[0]].rsplit('/')[-1].rsplit('.model.hdf5')[0]
    score_path = run_dir+'/test/scores/%s.score.json'%model_base
    if os.path.exists(score_path):
        with open(score_path,'r') as sf: score_data = json.load(sf)
    else:
        score_path = run_dir+'/scores/%s.score.json'%model_base
        with open(score_path,'r') as sf: score_data = json.load(sf)

    #better testing frame work for iteration...
    zoom_models = {'CNN':run_dir+'/models/L0-%s-LR.model.hdf5'%sv}

    if os.path.exists(base_dir+'/sms.maxima.json'):
        with open(base_dir+'/sms.maxima.json','r') as f: mask = json.load(f)
    if os.path.exists(base_dir+'/vcf/sms.maxima.json'):
        with open(base_dir+'/vcf/sms.maxima.json','r') as f: mask = json.load(f)
    with tf.device('/gpu:%s'%gpu_num):
        for sm in sorted(samples):
            s_start = time.time()
            print('::::::::::::::::::::::::::::::::::::::starting %s :::::::::::::::::::::::::::::::::::::::::::::::'%samples[sm].rsplit('/')[-1])
            f    = File(samples[sm],'r')
            sm   = list(f.keys())[0]
            rg   = list(f[sm].keys())[0]
            seqs = sorted(list(f[sm][rg].keys()),key=lambda x: x.zfill(255))
            if os.path.exists(base_dir+'/targets/%s.pickle.gz'%sm):
                vc_path   = base_dir+'/targets/%s.pickle.gz'%sm
            elif os.path.exists(base_dir+'/targets/%s_T.pickle.gz'%sm):
                vc_path   = base_dir+'/targets/%s_T.pickle.gz'%sm
            if os.path.exists(glob.glob(base_dir+'/comp/%s*.pickle.gz'%sm)[0]):
                comp_path = glob.glob(base_dir+'/comp/%s*.pickle.gz'%sm)[0]
            elif os.path.exists(glob.glob(base_dir+'/comp/%s_T*.pickle.gz'%sm)[0]):
                comp_path = glob.glob(base_dir+'/comp/%s_T*.pickle.gz'%sm)[0]
            else:
                comp_path = ''
            with gzip.GzipFile(vc_path,'rb') as vf:
                D = pickle.load(vf)
                T = {}
                for d in D[sv]:
                    if selected_seqs=='all' or d[0] in selected_seqs:
                        if sv=='INS': d[1],d[2],d[3] = d[1]-500,d[2]+500,abs(d[2]-d[1]+1000)
                        if d[3]>=0: # [0-100]=50 to [100-1000]=500 with midpoint = 50 + 450/2 = [275]
                            if d[0] in T: T[d[0]] += [d]
                            else:         T[d[0]]  = [d]
                for k in T: T[k] = sorted(T[k],key=lambda x: (x[0].zfill(255),x[1]))
            if os.path.exists(comp_path):
                with gzip.GzipFile(comp_path,'rb') as vf:
                    D = pickle.load(vf)
                    if 'type' in D: D = D['type']
                    if sv in D: #do a partial match of sv INV:PERFECT -> INV
                        Q = {}
                        for a_id in D[sv]:
                            d = D[sv][a_id][sm]
                            if sv=='INS': d[1],d[2],d[3] = d[1]-500,d[2]+500,abs(d[2]-d[1]+1000)
                            if d[3]>=0:
                                if d[0] in Q: Q[d[0]] += [d]
                                else:         Q[d[0]]  = [d]
                        for k in Q: Q[k] = sorted(Q[k],key=lambda x: (x[0].zfill(255),x[1]))
                    elif sv.rsplit(':')[0] in D:
                        sv_super = sv.rsplit(':')[0]
                        Q = {}
                        for a_id in D[sv_super]:
                            d = D[sv_super][a_id][sm]
                            if sv_super=='INS': d[1],d[2],d[3] = d[1]-500,d[2]+500,abs(d[2]-d[1]+1000)
                            if d[3]>=0:
                                if d[0] in Q: Q[d[0]] += [d]
                                else:         Q[d[0]]  = [d]
                        for k in Q: Q[k] = sorted(Q[k],key=lambda x: (x[0].zfill(255),x[1]))
                    else:
                        Q = {}
                        for seq in T: Q[seq] = []
            else:
                Q = {}
                for seq in T: Q[seq] = []
            #vc is [seq,start,end,svl,ht,naf,cn,a_id]
            print('read targets information')
            N,V,bw,zws = {},{'five':{'L,R':[],'LR':[]},'zoom':{}},25,[50]
            for seq in sorted(T)[::-1]:
                seq_start = time.time()
                if seq in f[sm][rg]: #check to see if it was processed
                    start = time.time()

                    labels = score_data['params']['labels']
                    base_score = float(score_data['score'])
                    geno_bins = []
                    for l in labels:
                        sv_type = l.rsplit('.')[0]
                        geno    = float('.'.join(l.rsplit('.')[1:]))
                        if sv_type!='NOT': geno_bins += [geno]
                    geno_bins = sorted(geno_bins)
                    batch_size,classes = 4096,len(labels)
                    #works with the five norm or zoom norm files
                    if type(f[sm][rg][seq]) is h5py._hl.dataset.Dataset:
                        ms      = f[sm][rg][seq].attrs['m']
                        ftr_idx = get_ftr_idx(f[sm][rg][seq])
                        th      = get_trans_hist(f[sm][rg][seq],ms)
                        data    = np.zeros(f[sm][rg][seq].shape,dtype=np.float32)
                        data[:] = f[sm][rg][seq][:]
                    else:
                        ms      = f[sm][rg][seq][str(bw)].attrs['m']
                        ftr_idx = get_ftr_idx(f[sm][rg][seq][str(bw)])
                        th      = get_trans_hist(f[sm][rg][seq][str(bw)],ms)
                        data    = np.zeros(f[sm][rg][seq][str(bw)].shape,dtype=np.float32)
                        data[:] = f[sm][rg][seq][str(bw)][:]
                    stop = time.time()
                    print('loaded sm=%s, rg=%s, seq=%s data in %s sec'%(sm,rg,seq,round(stop-start,2)))
                    print('checking w=%s---------------------------------------------------------------'%bw)
                    #model scanner
                    start  = time.time()
                    sv_mask = [[x[0]//bw,x[1]//bw] for x in mask[seq]]
                    M = {}
                    for mdl in five_models:
                        if mdl !='LR': M[mdl] = keras.models.load_model(five_models[mdl])
                    pred = predict_and_pack(data,M,sv_mask)
                    stop  = time.time()
                    print('scanned models=%s and applied mask regions in %s sec'%(M.keys(),round(stop-start,2)))
                    #:::::::CLUSTER:::PREDICTIONS:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                    start = time.time() #scan the likelihood estimates for L and R breakpoint ranges
                    order = {'L':0,'R':1}
                    hs    = {}
                    for i in range(1,classes,1): hs[i] = sorted(geno_bins)[i-1]
                    if sv=='DEL':   flank,max_w,k_c = 1.0,100,5
                    elif sv=='DUP': flank,max_w,k_c = 2.0,200,3
                    elif sv=='INV': flank,max_w,k_c = 0.5,100,3
                    else:           flank,max_w,k_c = 0.5,100,3
                    min_p = 0.5
                    B = scan_breaks(pred,m_labels=list(M.keys()),classes=classes)
                    C = cluster_knn_pairs(B,pred,sv_mask,k=k_c,order=order,trim=min_p,min_p=base_score,verbose=False)
                    if len(C)<=0:
                        print('no clustered pairs found for sv=%s on seq=%s'%(sv,seq))
                        K,KS = [],[]
                    elif not all([len(C[sorted(C)[0]][brk])>0 for brk in C[sorted(C)[0]]]):
                        while not all([len(C[sorted(C)[0]][brk])>0 for brk in C[sorted(C)[0]]]) and base_score>=min_p:
                            base_score-=0.01
                            C = cluster_knn_pairs(B,pred,sv_mask,k=k_c,order=order,trim=min_p,min_p=base_score,verbose=False)
                        if not all([len(C[sorted(C)[0]][brk])>0 for brk in C[sorted(C)[0]]]):
                            K,KS = [],[]
                        else:
                            K = cluster_calls(C,seq,pred,classes=classes,m_map=order,hs=hs,w=bw,n=11)
                            KS = sorted(K,key=lambda x: x[5])[::-1][0:int(0.5+min(5.0*len(T[seq]),len(K)))]
                    else:
                        K = cluster_calls(C,seq,pred,classes=classes,m_map=order,hs=hs,w=bw,n=11)
                        KS = sorted(K,key=lambda x: x[5])[::-1][0:int(0.5+min(5.0*len(T[seq]),len(K)))]
                    if sv=='INS':
                        for i in range(len(KS)):
                            KS[i][1] = KS[i][1]-500
                            KS[i][2] = KS[i][2]+500
                            KS[i][3] = KS[i][3]+1000
                    V['five']['L,R'] += KS
                else:
                    print('tensor was not processed for seq=%s score=%s'%(seq,sv_score(T[seq],[])))

                # print('starting %s five-L,R(%s)=>five-LR(%s) searches -------------------------------------------'%(len(K),bw,bw))
                # start = time.time()
                # MLR = {'LR':keras.models.load_model(five_models['LR'])}
                # lr_order = {'LR':0}
                # pred  = predict_and_pack(data,MLR,sv_mask)
                # lr_B     = scan_breaks(pred,m_labels=list(MLR.keys()),classes=classes)
                # lr_K     = lr_breaks_to_calls(lr_B[0],seq,pred,min_p=base_score,w=bw,n=11)
                # ############################################################################################################
                # PS = pred_from_vca(data,
                #                    five_models,
                #                    K,
                #                    hs=hs,w=bw,flank=flank,max_w=max_w,verbose=False)[0]
                # KF = sorted(PS,key=lambda x: x[5])[::-1][0:int(0.5+min(5.0*len(T[seq]),len(PS)))]
                # print('starting %s five-LR(%s)=>five-LR(%s) searches -------------------------------------------'%(len(lr_K),bw,bw))
                # lr_PS = pred_from_vca(data,
                #                       five_models,
                #                       lr_K,
                #                       hs=hs,w=bw,flank=flank,max_w=max_w,verbose=False)[0]
                # lr_KF = sorted(lr_PS,key=lambda x: x[5])[::-1][0:int(0.5+min(5.0*len(T[seq]),len(lr_PS)))]
                #
                # V['five']['LR'] += KF
                V['five']['LR'] = []
                KF = []
                lr_KF = []
                # stop = time.time()
                # print('L,R=>LR fitting time was %s sec' % round(stop-start,2))

                #scoring and merging........................
                if seq in Q:
                    print('comp score=%s'%sv_score(T[seq],Q[seq]))
                else:
                    print('comp score=%s'%sv_score(T[seq],[]))
                print('L,R  score=%s'%sv_score(T[seq],KS))
                print('L,R=>LR score=%s'%sv_score(T[seq],KF))
                print('LR=>LR score=%s'%sv_score(T[seq],lr_KF))
                seq_stop = time.time()
                print('seq=%s was processed in %s sec'%(seq,round(seq_stop-seq_start,2)))
            f.close()

            o_score1 = opt_score(V['five']['L,R'],T,metric='f1')
            o_score2 = opt_score(V['five']['LR'],T,metric='f1')
            if o_score1>o_score2:
                o_score,W = o_score1,V['five']['L,R']
                print('L,R=%s performed better than LR=%s'%(V['five']['L,R'],V['five']['LR']))
            else:
                o_score,W = o_score2,V['five']['LR']
                print('LR=%s performed better than L,R=%s'%(V['five']['LR'],V['five']['L,R']))

            #score the results-----------------------------------------------------------------------
            target = {}
            for row in W:
                if row[0] in T and not row[0] in target: target[row[0]] = int(round(1.0*len(T[row[0]])))
            ts = {}
            for seq in T:
                if seq in target: ts[seq] = T[seq]
            qs = {}
            for seq in Q:
                if seq in target: qs[seq] = Q[seq]
            SV = merge_calls(W,r=0.75,strategy='like',target=target)

            vca = []
            for seq in SV: vca += SV[seq]
            vca = sorted(vca,key=lambda x: (x[0].zfill(255),x[1]))

            print('comp score=%s' % sv_score(ts,qs))
            print('tensor score=%s'%sv_score(ts,SV))

            print('saving pickled results...')
            if not os.path.exists(out_dir): os.mkdir(out_dir)
            with gzip.GzipFile(out_dir+'/%s.%s.tensorsv.pickle.gz'%(sm,sv),'wb') as f:
                D = {sm:{sv:{'vca':{'tensor':W,'ts':ts,'comp':qs},'score':sv_score(ts,SV),'opt_score':o_score}}}
                pickle.dump(D,f)
                print(True)