#!/usr/env/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals #python 3.6+
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore',category=Warning)
import time
import argparse
import os
import json
import shutil
import glob
import pickle
import gzip
import subprocess
import numpy as np
import tensorflow as tf
from h5py import File
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e: print(e)

class DataGen(tf.keras.utils.Sequence):
    def __init__(self,data_path,sms_list,labels={'NOT.1.0':0,'DEL.0.5':1,'DEL.1.0':2},size_range=[0,int(3E8)],
                 sub_sample=1.0,sub_sample_not=1.0,balance_not=True,cols=[0,55],ftrs=range(17),mmnts=range(4),
                 sample_random=True,sample_batch=8,batch=256,shuffle=True,verbose=False):
        self.data_path=data_path
        self.sms_list=sorted(sms_list)
        self.labels=labels
        self.size_range=size_range
        self.sub_sample=sub_sample
        self.sub_sample_not=sub_sample_not
        self.balance_not=balance_not
        self.cols=cols
        self.ftrs=ftrs
        self.mmnts=mmnts
        self.sample_batch=sample_batch
        self.sample_index=0
        self.sample_random=sample_random
        self.batch=batch
        self.shuffle=shuffle
        self.verbose=verbose
        self.step  = 0
        self.epoch = 0
        self.start = time.time()
        self.stop  = time.time()
        self.prep_data()

    def __len__(self):
        return int(np.ceil(len(self.X)/float(self.batch)))

    def __getitem__(self,idx):
        self.step += 1
        batch_x = self.X[idx*self.batch:(idx+1)*self.batch]
        batch_y = self.Y[idx*self.batch:(idx+1)*self.batch]
        if self.verbose: print('--==:: STEP %s ::==--'%self.step)
        return np.array(batch_x),np.array(batch_y)

    def on_epoch_end(self):
        self.step   = 0
        self.epoch += 1
        msg = '-----====||| EPOCH %s: %s tensor in %s sec |||====-----'
        print(msg%(str(self.epoch).rjust(2),str(self.X.shape).rjust(2),round(self.stop-self.start,2)))
        self.sample_index += self.sample_batch
        self.prep_data()

    def prep_data(self):
        self.start = time.time()
        X,Y,Z,T,z = [],[],{},{},0
        f     = File(self.data_path,'r')
        if self.sample_random:
            sms = sorted(list(np.random.choice(self.sms_list,min(len(self.sms_list),self.sample_batch),replace=False)))
        else:
            sms = self.sms_list[self.sample_index:(self.sample_index+min(self.sample_batch,len(self.sms_list)))]
        ftrs  = np.asarray(sorted(list(self.ftrs)))
        mmnts = np.asarray(sorted(list(self.mmnts)))
        for sm in f:
            if sm in sms:  #will skip to only those selected
                for k in f[sm]:
                    match,sv_type = False,k #will do extact or partial match of labels: NOT=>NOT.1.0 = match!
                    for l in self.labels:
                        if sv_type.find(l)>-1: match,k = True,l
                    if match:
                        #[1] calculate how many total are within the size_range: must use f[sm][k].attrs['svlen'], but not k='NOT.1.0'
                        if 'svlen' in f[sm][k].attrs:
                            size_idx,sv_lens = [],np.array(np.round(np.exp(f[sm][k].attrs['svlen']/12.75)),dtype=int)
                            for i in range(len(sv_lens)):
                                if sv_lens[i]>=self.size_range[0] and sv_lens[i]<self.size_range[1]:
                                    size_idx += [i]
                        else: size_idx = range(f[sm][k].shape[0])
                        if len(size_idx)>0:
                            l,mulx   = len(size_idx),1.0
                            if self.sub_sample<1.0:                          mulx = max(0.0,min(1.0,self.sub_sample))
                            if k.find('NOT')>=0 and self.sub_sample_not<1.0: mulx = max(0.0,min(1.0,self.sub_sample_not))
                            set_a = int(round(mulx*len(size_idx)))
                            if len(size_idx)>0:
                                idx = sorted(list(np.random.choice(range(len(size_idx)),set_a,replace=False)))
                                X  += [row[self.cols[0]:self.cols[1]][:,ftrs,:][:,:,mmnts] for row in f[sm][k][idx]]
                                Y  += [[self.labels[k]] for row in f[sm][k][idx]]
                                if k in Z: Z[k] += [z+x for x in range(len(idx))]
                                else:      Z[k]  = [z+x for x in range(len(idx))]
                                z   += len(idx)
                                if k in T: T[k] += len(idx)
                                else:      T[k]  = len(idx)
                                if self.verbose:
                                    print('loaded %s label=%s sv_len=[%s,%s] for sm=%s'%(len(idx),k,self.size_range[0],self.size_range[1],sm))
        f.close()
        if self.balance_not: #option to make NOT labels equal in proportion to the sum of the others...
            balance_sums = 0
            for k in T:
                if k.find('NOT')<0: balance_sums += T[k]
            if self.verbose: print('balancing NOT labels from %s to %s'%(T,balance_sums))
            trim_idx = []
            for k in Z:
                if k.find('NOT')>=0: trim_idx += list(np.random.choice(Z[k],balance_sums,replace=False))
                else:                trim_idx += Z[k]
            trim_idx = sorted(trim_idx)
            XB,YB = [],[]
            for i in range(len(trim_idx)):
                XB += [X[trim_idx[i]]]
                YB += [Y[trim_idx[i]]]
            X,Y = XB,YB
        X,Y = np.asarray(X),np.asarray(Y,dtype=np.uint8)
        self.indices = np.arange(len(X))
        if self.shuffle: np.random.shuffle(self.indices)
        self.X,self.Y = X[self.indices],Y[self.indices]
        self.stop   = time.time()
        if verbose: print('Finished loading data for Epoch=%s'%(self.epoch+1))

#make a training, validation sample set, while excluding test_sms
def partition_samples(data_path,train_sms='all',test_sms=[],split=0.75):
    f = File(data_path,'r')
    if train_sms=='all': train_sms = list(f.keys())
    valid_sms = sorted(list(np.random.choice(train_sms,int((1.0-split)*len(train_sms)),replace=False)))
    train_sms = sorted(list(set(train_sms).difference(set(valid_sms))))
    return train_sms,valid_sms

#use for both training and test proceedures, sample_batch_size limits the number of loaded samples
#takes normalized, tensor data with an optional split=[0.0,1.0]=>set(A),set(B)
#returns the solid tensor block of labeled data, cols can select parts of th e[a,l,m,r,b] frames
#ftrs can select individual features, while mmnts can pick m1,m2,m3,m4 for any feature
#additional functionality allows types such as DEL,DUP,INV to be sub sampled at a rate
#that is independantly controlled versus the background :sub_sample_not
def data_gen(data_path,sms_list,labels={'NOT.1.0':0,'DEL.0.5':1,'DEL.1.0':2},size_range=[0,int(250E6)],
             sub_sample=1.0,sub_sample_not=1.0,balance_not=True,cols=[0,55],ftrs=range(17),mmnts=range(4),
             sample_batch=8,batch=256,verbose=False):
    while True:
        X,Y,Z,T,z = [],[],{},{},0
        f     = File(data_path,'r')
        sms   = sorted(list(np.random.choice(sms_list,sample_batch,replace=False)))
        ftrs  = np.asarray(sorted(list(ftrs)))
        mmnts = np.asarray(sorted(list(mmnts)))
        for sm in f:
            if sm in sms:  #will skip to only those selected
                for k in f[sm]:
                    match,sv_type = False,k #will do extact or partial match of labels: NOT=>NOT.1.0 = match!
                    for l in labels:
                        if sv_type.find(l)>-1: match,k = True,l
                    if match:
                        #[1] calculate how many total are within the size_range: must use f[sm][k].attrs['svlen'], but not k='NOT.1.0'
                        if 'svlen' in f[sm][k].attrs:
                            size_idx,sv_lens = [],np.array(np.round(np.exp(f[sm][k].attrs['svlen']/12.75)),dtype=int)
                            for i in range(len(sv_lens)):
                                if sv_lens[i]>=size_range[0] and sv_lens[i]<size_range[1]:
                                    size_idx += [i]
                        else: size_idx = range(f[sm][k].shape[0])
                        if len(size_idx)>0:
                            l,mulx   = len(size_idx),1.0
                            if sub_sample<1.0:                          mulx = max(0.0,min(1.0,sub_sample))
                            if k.find('NOT')>=0 and sub_sample_not<1.0: mulx = max(0.0,min(1.0,sub_sample_not))
                            set_a = int(round(mulx*len(size_idx)))
                            if len(size_idx)>0:
                                idx = sorted(list(np.random.choice(range(len(size_idx)),set_a,replace=False)))
                                X  += [row[cols[0]:cols[1]][:,ftrs,:][:,:,mmnts] for row in f[sm][k][idx]]
                                Y  += [[labels[k]] for row in f[sm][k][idx]]
                                if k in Z: Z[k] += [z+x for x in range(len(idx))]
                                else:         Z[k]  = [z+x for x in range(len(idx))]
                                z   += len(idx)
                                if k in T: T[k] += len(idx)
                                else:      T[k]  = len(idx)
                                if verbose: print('loaded %s label=%s sv_len=[%s,%s] for sm=%s'%(len(idx),k,size_range[0],size_range[1],sm))
        f.close()
        if balance_not: #option to make NOT labels equal in proportion to the sum of the others...
            balance_sums = 0
            for k in T:
                if k.find('NOT')<0: balance_sums += T[k]
            if verbose: print('balancing NOT labels from %s to %s'%(T,balance_sums))
            trim_idx = []
            for k in Z:
                if k.find('NOT')>=0: trim_idx += list(np.random.choice(Z[k],balance_sums,replace=False))
                else:                trim_idx += Z[k]
            trim_idx = sorted(trim_idx)
            XB,YB = [],[]
            for i in range(len(trim_idx)):
                XB += [X[trim_idx[i]]]
                YB += [Y[trim_idx[i]]]
            X,Y = XB,YB
        X,Y = np.asarray(X),np.asarray(Y,dtype=np.uint8)
        batch = min(batch,len(Y))
        batch_idx = np.random.choice(range(len(Y)),batch,replace=False)
        X,Y = X[batch_idx],Y[batch_idx]
        yield(X,Y)

#make data_load cachable to memory....
def data_load(data_in,sms_list,labels={'NOT.1.0':0,'DEL.0.5':1,'DEL.1.0':2},size_range=[0,int(250E6)],
              sub_sample=1.0,sub_sample_not=1.0,balance_not=True,counter_label=True,cols=[0,55],ftrs=range(17),mmnts=range(4),
              geno_cut=0.0,sample_batch=32,verbose=False):
        X,Y,Z,T,z = [],[],{},{},0
        if type(data_in) is str:
            f     = File(data_in,'r')
            sms   = sorted(list(np.random.choice(sms_list,min(len(sms_list),sample_batch),replace=False)))
            ftrs  = np.asarray(sorted(list(ftrs)))
            mmnts = np.asarray(sorted(list(mmnts)))
            for sm in f:
                if sm in sms:  #will skip to only those selected
                    for k in f[sm]:
                        sv_type,K = k.rsplit('.')[0],{} #will do extact or partial match of labels: NOT=>NOT.1.0 = match!
                        for l in labels:
                            if k==l or l.find(sv_type)>-1:
                                if k in K: K[k] += [l]
                                else:      K[k]  = [l]
                        if len(K)>0: #had a match------------------------------------------------------------------------------
                            geno_bins,h = {},float('.'.join(k.split('.')[1:]))
                            if h>=geno_cut:
                                if len(K[list(K.keys())[0]])>1:
                                    try:
                                        lbls = sorted(K[list(K.keys())[0]],key=lambda x: float('.'.join(x.rsplit('.')[1:])))
                                        min_l = [1.0,1]
                                        for i in range(len(lbls)):
                                            b = abs(h-float('.'.join(lbls[i].rsplit('.')[1:])))
                                            if b<min_l[0]: min_l = [b,i]
                                        l = lbls[min_l[1]]
                                    except Exception: pass
                                else: l = K[list(K.keys())[0]][0]
                                K[k] = l
                                #[1] calculate how many total are within the size_range: must use f[sm][k].attrs['svlen'], but not k='NOT.1.0'
                                if 'svlen' in f[sm][k].attrs:
                                    size_idx,sv_lens = [],np.array(np.round(np.exp(f[sm][k].attrs['svlen']/12.75)),dtype=int) #exp transform back
                                    for i in range(len(sv_lens)):
                                        if sv_lens[i]>=size_range[0] and sv_lens[i]<size_range[1]:
                                            size_idx += [i]
                                else: size_idx = range(f[sm][k].shape[0])
                                if len(size_idx)>0:
                                    mulx = 1.0
                                    if sub_sample < 1.0:                            mulx = max(0.0, min(1.0, sub_sample))
                                    if k.find('NOT') >= 0 and sub_sample_not < 1.0: mulx = max(0.0,min(1.0, sub_sample_not))
                                    set_a = int(round(mulx*len(size_idx)))
                                    if len(size_idx)>0:
                                        idx = sorted(list(np.random.choice(range(len(size_idx)),set_a,replace=False)))
                                        X  += [row[cols[0]:cols[1]][:,ftrs,:][:,:,mmnts] for row in f[sm][k][idx]]
                                        Y  += [[labels[l]] for row in f[sm][k][idx]]
                                        if k in Z: Z[k] += [z+x for x in range(len(idx))]
                                        else:      Z[k]  = [z+x for x in range(len(idx))]
                                        z   += len(idx)
                                        if k in T: T[k] += len(idx)
                                        else:      T[k]  = len(idx)
                                        if verbose: print('loaded %s label=%s sv_len=[%s,%s] for sm=%s'%(len(idx),k,size_range[0],size_range[1],sm))
                            else:
                                print('filtered %s label=%s sv_len=[%s,%s] for sm=%s'%(len(f[sm][k]['tensor']),k,size_range[0],size_range[1],sm))
                        elif counter_label: #counter labels get added to the NOT.1.0 label and then can be balanced
                            if 'svlen' in f[sm][k].attrs:
                                size_idx,sv_lens = [],np.array(np.round(np.exp(f[sm][k].attrs['svlen']/12.75)),dtype=int) #exp transform back
                                for i in range(len(sv_lens)):
                                    if sv_lens[i]>=size_range[0] and sv_lens[i]<size_range[1]:
                                        size_idx += [i]
                            else: size_idx = range(f[sm][k].shape[0])
                            if len(size_idx)>0:
                                mulx = 1.0
                                if sub_sample_not < 1.0: mulx = max(0.0,min(0.5,sub_sample_not))
                                set_a = int(round(mulx*len(size_idx)))
                                if len(size_idx)>0:
                                    idx = sorted(list(np.random.choice(range(len(size_idx)),set_a,replace=False)))
                                    X  += [row[cols[0]:cols[1]][:,ftrs,:][:,:,mmnts] for row in f[sm][k][idx]]
                                    Y  += [[0] for row in f[sm][k][idx]] #k is not the label you wanted IE will get the 0=>'NOT.1.0' label
                                    if k in Z: Z[k] += [z+x for x in range(len(idx))]
                                    else:      Z[k]  = [z+x for x in range(len(idx))]
                                    z   += len(idx)
                                    if k in T: T[k] += len(idx)
                                    else:      T[k]  = len(idx)
                                    if verbose: print('loaded %s counter_label=%s sv_len=[%s,%s] for sm=%s'%(len(idx),k,size_range[0],size_range[1],sm))
            f.close() #read from disk
        elif type(data_in) is dict: #cached data load
            f     = data_in
            sms   = sorted(list(np.random.choice(sms_list,min(len(sms_list),sample_batch),replace=False)))
            ftrs  = np.asarray(sorted(list(ftrs)))
            mmnts = np.asarray(sorted(list(mmnts)))
            for sm in f:
                if sm in sms:  #will skip to only those selected
                    for k in f[sm]:
                        sv_type,K = k.rsplit('.')[0],{} #will do extact or partial match of labels: NOT=>NOT.1.0 = match!
                        for l in labels:
                            if k==l or l.find(sv_type)>=0:
                                if k in K: K[k] += [l]
                                else:      K[k]  = [l]
                        if len(K)>0: #had a match------------------------------------------------------------------------------
                            geno_bins,h = {},float('.'.join(k.split('.')[1:]))
                            if h>=geno_cut:
                                if len(K[list(K.keys())[0]])>1:
                                    try:
                                        lbls = sorted(K[list(K.keys())[0]],key=lambda x: float('.'.join(x.rsplit('.')[1:])))
                                        min_l = [1.0,1]
                                        for i in range(len(lbls)):
                                            b = abs(h-float('.'.join(lbls[i].rsplit('.')[1:])))
                                            if b<min_l[0]: min_l = [b,i]
                                        l = lbls[min_l[1]]
                                    except Exception: pass
                                else: l = K[list(K.keys())[0]][0]
                                K[k] = l
                                #[1] calculate how many total are within the size_range: must use f[sm][k].attrs['svlen'], but not k='NOT.1.0'
                                if 'svlen' in f[sm][k]:
                                    size_idx,sv_lens = [],f[sm][k]['svlen'] #cached data has been exp tranformed back already...
                                    for i in range(len(sv_lens)):
                                        if sv_lens[i]>=size_range[0] and sv_lens[i]<size_range[1]:
                                            size_idx += [i]
                                else: size_idx = range(f[sm][k]['tensor'].shape[0])
                                if len(size_idx)>0:
                                    mulx   = 1.0
                                    if sub_sample<1.0:                          mulx = max(0.0,min(1.0,sub_sample))
                                    if k.find('NOT')>=0 and sub_sample_not<1.0: mulx = max(0.0,min(1.0,sub_sample_not))
                                    set_a = int(round(mulx*len(size_idx)))
                                    if len(size_idx)>0:
                                        idx = sorted(list(np.random.choice(range(len(size_idx)),set_a,replace=False)))
                                        X  += [row[cols[0]:cols[1]][:,ftrs,:][:,:,mmnts] for row in f[sm][k]['tensor'][idx]]
                                        Y  += [[labels[l]] for row in f[sm][k]['tensor'][idx]]
                                        if k in Z:    Z[k] += [z+x for x in range(len(idx))]
                                        else:         Z[k]  = [z+x for x in range(len(idx))]
                                        z   += len(idx)
                                        if k in T: T[k] += len(idx)
                                        else:      T[k]  = len(idx)
                                        if verbose: print('loaded %s label=%s=>%s sv_len=[%s,%s] for sm=%s'%(len(idx),k,l,size_range[0],size_range[1],sm))
                            else:
                                print('filtered %s label=%s sv_len=[%s,%s] for sm=%s'%(len(f[sm][k]['tensor']),k,size_range[0],size_range[1],sm))
                        elif counter_label: #counter labels get added to the NOT.1.0 label and then can be balanced
                            if 'svlen' in f[sm][k]:
                                size_idx,sv_lens = [],f[sm][k]['svlen'] #cached data has been exp tranformed back already...
                                for i in range(len(sv_lens)):
                                    if sv_lens[i]>=size_range[0] and sv_lens[i]<size_range[1]:
                                        size_idx += [i]
                            else: size_idx = range(f[sm][k]['tensor'].shape[0])
                            if len(size_idx)>0:
                                mulx = 1.0
                                if sub_sample_not < 1.0: mulx = max(0.0,min(0.5,sub_sample_not))
                                set_a = int(round(mulx*len(size_idx)))
                                if len(size_idx)>0:
                                    idx = sorted(list(np.random.choice(range(len(size_idx)),set_a,replace=False)))
                                    X  += [row[cols[0]:cols[1]][:,ftrs,:][:,:,mmnts] for row in f[sm][k]['tensor'][idx]]
                                    Y  += [[0] for row in f[sm][k]['tensor'][idx]] #k is not the label you wanted IE will get the 0=>'NOT.1.0' label
                                    if k in Z: Z[k] += [z+x for x in range(len(idx))]
                                    else:      Z[k]  = [z+x for x in range(len(idx))]
                                    z   += len(idx)
                                    if k in T: T[k] += len(idx)
                                    else:      T[k]  = len(idx)
                                    if verbose: print('loaded %s counter_label=%s sv_len=[%s,%s] for sm=%s'%(len(idx),k,size_range[0],size_range[1],sm))
        if balance_not: #option to make NOT labels equal in proportion to the sum of the others...
            balance_sums = 0
            for k in T:
                if k.find('NOT')<0: balance_sums += T[k]
            if verbose: print('balancing NOT labels from %s to %s'%(T,balance_sums))
            trim_idx = []
            for k in Z:
                if k.find('NOT')>=0: trim_idx += list(np.random.choice(Z[k],balance_sums,replace=False))
                else:                trim_idx += Z[k]
            trim_idx = sorted(trim_idx)
            XB,YB = [],[]
            for i in range(len(trim_idx)):
                XB += [X[trim_idx[i]]]
                YB += [Y[trim_idx[i]]]
            X,Y = XB,YB
        X,Y = np.asarray(X),np.asarray(Y,dtype=np.uint8)
        return X,Y

#read a tensor and cache to main memory
def data_cache(data_path,sms):
    data_in = {}
    f = File(data_path,'r')
    for sm in f:
        if sm in sms:
            data_in[sm] = {}
            for k in f[sm]:
                data_in[sm][k] = {}
                if 'svlen' in f[sm][k].attrs:
                    data_in[sm][k] = {'svlen':np.array(np.round(np.exp(f[sm][k].attrs['svlen']/12.75)),dtype=int)} #NOTs are too large to save in hdf5 attrs...
                data_in[sm][k]['tensor'] = f[sm][k][:]
    f.close()
    return data_in

def confusion_matrix(pred_labels,test_labels,classes):
    M = {}
    for i in range(classes):
        for j in range(classes):
            M[(i,j)] = 0.0
    for i in range(len(test_labels)):
        M[(test_labels[i],pred_labels[i])] += 1.0
    return M

def metrics(M):
    ls,P,R,F1 = set([]),{},{},{}
    for i,j in M:
        if i in P: P[i] += [M[(i,j)]]
        else:      P[i]  = [M[(i,j)]]
        if j in R: R[j] += [M[(i,j)]]
        else:      R[j]  = [M[(i,j)]]
        ls.add(i)
        ls.add(j)
    for l in ls:
        sum_p = sum(P[l])
        if sum_p>0.0:     P[l]  = M[(l,l)]/sum(P[l])
        else:             P[l]  = 0.0
        sum_r = sum(R[l])
        if sum_r>0.0:     R[l]  = M[(l,l)]/sum(R[l])
        else:             R[l]  = 0.0
        if P[l]+R[l]>0.0: F1[l] = 2.0*(P[l]*R[l])/(P[l]+R[l])
        else:             F1[l] = 0.0
    return P,R,F1

def plot_train_test(history,title,ylim=[0.0,1.0],out_path=None,font_size=10):
    if 'acc' in history:            plt.plot(history['acc'])
    elif 'accuracy' in history:     plt.plot(history['accuracy'])
    if 'val_acc' in history:        plt.plot(history['val_acc'])
    elif 'val_accuracy' in history: plt.plot(history['val_accuracy'])
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    axes = plt.gca()
    axes.set_ylim(ylim)
    plt.title(title)
    plt.ylabel('Accuracy & Loss Value')
    plt.xlabel('Epoch')
    plt.legend(['TRN-ACC','VAL-ACC','TRN-LOSS','VAL-LOSS'], loc='lower left')
    plt.rcParams.update({'font.size': font_size})
    if out_path is not None: plt.savefig(out_path); plt.close()
    else: plt.show()
    return True

def plot_confusion_heatmap(confusion_matrix,title,offset=0,out_path=None,font_size=10):
    plt.rcParams.update({'font.size': font_size})
    xs = set([])
    for i,j in confusion_matrix:
        xs.add(i+offset)
        xs.add(j+offset)
    sx = sorted(list(xs))
    h = np.zeros((len(sx),len(sx)),dtype=float)
    for i,j in confusion_matrix: h[i,j] = confusion_matrix[(i,j)]
    for i in range(len(h)): h[i] /= sum(h[i])
    plt.imshow(h,cmap='Greys')
    plt.xticks(range(len(sx)),sx)
    plt.yticks(range(len(sx)),sx)
    plt.title(title)
    plt.ylabel('Test Class')
    plt.xlabel('Pred Class')
    plt.colorbar()
    plt.rcParams.update({'font.size': font_size})
    if out_path is not None: plt.savefig(out_path); plt.close()
    else: plt.show()
    return True

#optional model plotting
def plot_model(model,out_path):
    tf.keras.utils.plot_model(
        model, to_file=out_path+'/model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=300
    )

def build_model(dims,classes,cmx,decay,kf=5,drop=0.25,levels=2,form='cnn'):
    #----------------------------------------------------------------------------------------------
    if form=='cnn':
        model = tf.keras.Sequential()
        #(start)..................................................................................................................
        model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                         padding='same',activation='relu',input_shape=dims,
                                         kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
        if levels>1:
            model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',
                                             padding='same',kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
        if model.output_shape[1]//4>1: w_pool = 4
        else:                          w_pool = 1
        if model.output_shape[2]//2>1: h_pool = 2
        else:                          h_pool = 1
        if w_pool>1 or h_pool>1:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(w_pool,h_pool)))
        model.add(tf.keras.layers.Dropout(drop))
        # (start)................................................................................................................
        if levels>2:
            model.add(tf.keras.layers.Conv2D(cmx*4,(min(kf+2,dims[1]),min(kf+2,dims[2])),activation='relu',padding='same',
                                             kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
            if levels>3:
                model.add(tf.keras.layers.Conv2D(cmx*4,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
            if model.output_shape[1]//2>1: w_pool = 2
            else:                          w_pool = 1
            h_pool = 1
            if w_pool > 1 or h_pool > 1:
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(w_pool, h_pool)))
            model.add(tf.keras.layers.Dropout(drop))
            if levels>4:
                model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                           kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                if levels>5:
                    model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                model.add(tf.keras.layers.Dropout(drop*2.0))
                if levels>6:
                    model.add(tf.keras.layers.Conv2D(cmx*4,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                               kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                    if levels>8:
                        model.add(tf.keras.layers.Conv2D(cmx*4,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                                   kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                    model.add(tf.keras.layers.Dropout(drop*2.0))
                    if levels>9:
                        model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                                   kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                        if levels>10:
                            model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                        model.add(tf.keras.layers.Dropout(drop*2.0))
                        if levels>11:
                            model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                                   kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                            if levels>12:
                                model.add(tf.keras.layers.Conv2D(cmx*2,(min(kf,dims[1]),min(kf,dims[2])),activation='relu',padding='same',
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
                            model.add(tf.keras.layers.Dropout(drop*2.0))
        # (end)...........................................................................
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(cmx*32,activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l=decay)))
        model.add(tf.keras.layers.Dropout(drop/2.0))
        model.add(tf.keras.layers.Dense(classes,activation='softmax',dtype=np.float32))
    elif form=='a_cnn':
        seq_in = tf.keras.layers.Input(shape=dims,name='seq_in')
        #----------------------------------------------------------------------------------
        conv1 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv1')(seq_in)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(max(2,min(kf,dims[1]//2)),1),name='pool1')(conv1)
        drop1 = drop_f = tf.keras.layers.Dropout(drop,name='drop_1')(pool1)
        conv2 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv2')(conv1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(max(2,min(kf,dims[1]//2)),1),name='pool2')(conv2)
        drop2 = drop_f = tf.keras.layers.Dropout(drop,name='drop_2')(pool2)
        att1        = tf.keras.layers.Attention(name='att1')([drop1,drop2]) #Luong-style
        q_gru_att_1 = tf.keras.layers.GlobalAveragePooling2D()(att1)
        q_gru_1     = tf.keras.layers.GlobalAveragePooling2D()(conv1)
        cnt_1       = tf.keras.layers.Concatenate()([q_gru_1, q_gru_att_1])
        #------------------------------------------------------------------------------------
        conv3 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv3')(conv2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(max(2,min(kf,dims[1]//2)),1),name='pool3')(conv3)
        drop3 = drop_f = tf.keras.layers.Dropout(drop,name='drop_3')(pool3)
        conv4 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv4')(conv3)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(max(2,min(kf,dims[1]//2)),1),name='pool4')(conv4)
        drop4 = drop_f = tf.keras.layers.Dropout(drop,name='drop_4')(pool4)
        att2        = tf.keras.layers.Attention(name='att2')([drop3,drop4])
        q_gru_att_2 = tf.keras.layers.GlobalAveragePooling2D()(att2)
        q_gru_2     = tf.keras.layers.GlobalAveragePooling2D()(conv3)
        cnt_2       = tf.keras.layers.Concatenate()([q_gru_2, q_gru_att_2])
        #-------------------------------------------------------------------------------------
        cnt         = tf.keras.layers.Concatenate()([cnt_1,cnt_2])
        #---------------------------------------------------------------------------------------------------------------
        dense_f = tf.keras.layers.Dense(32*cmx, activation="relu",name='dense_f')(cnt)
        drop_f = tf.keras.layers.Dropout(2*drop,name='drop_f')(dense_f)
        out_class = tf.keras.layers.Dense(classes,activation="softmax",dtype=np.float32,name='out_class')(drop_f)
        model = tf.keras.Model(inputs=seq_in, outputs=out_class)
    elif form=='a_gru_cnn':
        seq_in = tf.keras.layers.Input(shape=dims,name='seq_in')
        conv1  = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                        padding='same',activation='relu',input_shape=dims,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                        name='conv1')(seq_in)
        drop1  = tf.keras.layers.Dropout(drop,name='drop1')(conv1)
        dense1 = tf.keras.layers.Dense(1,activation="relu",name='dense1')(drop1)
        drop2  = tf.keras.layers.Dropout(drop,name='drop2')(dense1)
        features = tf.keras.layers.Reshape((dims[0],dims[1]),
                                           input_shape=drop1.shape,name='features')(drop2)
        #---------------------------------------------------------------------------------------------------------------
        gru1,state_h1 = tf.keras.layers.GRU(cmx,
                                            return_sequences=True,
                                            return_state=True,
                                            name='gru1')(features)
        gru2,state_h2 = tf.keras.layers.GRU(cmx,
                                            return_sequences=True,
                                            return_state=True,
                                            go_backwards=False,
                                            name='gru2')(gru1)
        att = tf.keras.layers.Attention(name='att1')([gru1,gru2])
        query_gru = tf.keras.layers.GlobalAveragePooling1D()(gru1)
        query_gru_att = tf.keras.layers.GlobalAveragePooling1D()(att)
        context = tf.keras.layers.Concatenate()([query_gru, query_gru_att])
        #---------------------------------------------------------------------------------------------------------------
        dense2 = tf.keras.layers.Dense(32*cmx, activation="relu",name='dense2')(context)
        drop4 = tf.keras.layers.Dropout(2*drop,name='drop4')(dense2)
        out1 = tf.keras.layers.Dense(classes,activation="softmax",dtype=np.float32,name='class_out')(drop4)
        model = tf.keras.Model(inputs=seq_in, outputs=out1)
    elif form=='a_bgru_cnn':
        seq_in = tf.keras.layers.Input(shape=dims,name='seq_in')
        conv1 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv1')(seq_in)
        drop1  = tf.keras.layers.Dropout(drop,name='drop1')(conv1)
        dense1 = tf.keras.layers.Dense(1,activation="relu",name='dense1')(drop1)
        drop2  = tf.keras.layers.Dropout(drop,name='drop2')(dense1)
        features = tf.keras.layers.Reshape((dims[0],dims[1]),
                                           input_shape=drop1.shape,name='features')(drop2)
        #---------------------------------------------------------------------------------------------------------------
        gru1,state_h1 = tf.keras.layers.GRU(cmx,
                                            return_sequences=True,
                                            return_state=True,
                                            name='gru1')(features)
        gru2,state_h2 = tf.keras.layers.GRU(cmx,
                                            return_sequences=True,
                                            return_state=True,
                                            go_backwards=True,
                                            name='gru2')(gru1)
        att = tf.keras.layers.Attention(name='att1')([gru1,gru2])
        query_gru = tf.keras.layers.GlobalAveragePooling1D()(gru1)
        query_gru_att = tf.keras.layers.GlobalAveragePooling1D()(att)
        context = tf.keras.layers.Concatenate()([query_gru, query_gru_att])
        #---------------------------------------------------------------------------------------------------------------
        dense2 = tf.keras.layers.Dense(32*cmx, activation="relu",name='dense2')(context)
        drop4 = tf.keras.layers.Dropout(2*drop,name='drop4')(dense2)
        out1 = tf.keras.layers.Dense(classes,activation="softmax",dtype=np.float32,name='class_out')(drop4)
        model = tf.keras.Model(inputs=seq_in, outputs=out1)
    elif form=='gru_cnn':
        seq_in = tf.keras.layers.Input(shape=dims,name='seq_in')
        conv1 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv1')(seq_in)
        drop1  = tf.keras.layers.Dropout(drop,name='drop1')(conv1)
        dense1 = tf.keras.layers.Dense(1,activation="relu",name='dense1')(drop1)
        drop2  = tf.keras.layers.Dropout(drop,name='drop2')(dense1)
        features = tf.keras.layers.Reshape((dims[0],dims[1]),
                                           input_shape=drop1.shape,name='features')(drop2)
        #---------------------------------------------------------------------------------------------------------------
        gru1 = tf.keras.layers.GRU(2*cmx,name='gru1')(features)
        #---------------------------------------------------------------------------------------------------------------
        dense2 = tf.keras.layers.Dense(32*cmx, activation="relu",name='dense2')(gru1)
        drop4 = tf.keras.layers.Dropout(2*drop,name='drop4')(dense2)
        out1 = tf.keras.layers.Dense(classes,activation="softmax",dtype=np.float32,name='class_out')(drop4)
        model = tf.keras.Model(inputs=seq_in, outputs=out1)
    elif form=='bgru_cnn':
        seq_in = tf.keras.layers.Input(shape=dims,name='seq_in')
        conv1 = tf.keras.layers.Conv2D(2*cmx,(min(kf+2,dims[1]),min(kf+2,dims[2])),
                                       padding='same',activation='relu',input_shape=dims,
                                       kernel_regularizer=tf.keras.regularizers.l2(l=decay),
                                       name='conv1')(seq_in)
        drop1  = tf.keras.layers.Dropout(drop,name='drop1')(conv1)
        dense1 = tf.keras.layers.Dense(1,activation="relu",name='dense1')(drop1)
        drop2  = tf.keras.layers.Dropout(drop,name='drop2')(dense1)
        features = tf.keras.layers.Reshape((dims[0],dims[1]),input_shape=drop1.shape,name='features')(drop2)
        #---------------------------------------------------------------------------------------------------------------
        bgru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(cmx),name='bgru1')(features)
        #---------------------------------------------------------------------------------------------------------------
        dense2 = tf.keras.layers.Dense(32*cmx, activation="relu",name='dense2')(bgru1)
        drop4 = tf.keras.layers.Dropout(2*drop,name='drop4')(dense2)
        out1 = tf.keras.layers.Dense(classes,activation="softmax",name='class_out')(drop4)
        model = tf.keras.Model(inputs=seq_in, outputs=out1)
    # (end)...........................................................................
    #----------------------------------------------------------------------------------------------
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])
    return model

if __name__ == '__main__':
    des = """train_sv: TensorSV Model Training Tool v0.1.4\nCopyright (C) 2020-2021 Timothy James Becker\n"""
    parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_path',type=str,help='input path to labeled input tensors: all.label.hdf5 data\t[None]')
    parser.add_argument('--out_path',type=str,help='output path where models, plots and scores will be stored\t[None]')
    parser.add_argument('--sv_types',type=str,help='comma-seperated DEL,DUP,CNV,INV,INS,TRA svtypes\t[DEL]')
    parser.add_argument('--geno_bins',type=str,help='comma-seperated upper bin boundries for each sv\t[0.5,1.0]')
    parser.add_argument('--no_geno',action='store_true',help='binary models instead of terniary: 1/1 instead of 0/1 on calls\t[False]')
    parser.add_argument('--geno_cut',type=float,help='genotype fraction cutoff of labeled input tensors\t[0.0]')
    parser.add_argument('--short_test',action='store_true',help='do not use longer whole genome sequence testing\t[False]')
    parser.add_argument('--balance',action='store_true',help='balance background number to sum of genotyped sv_type number\t[False]')
    parser.add_argument('--counter_label',action='store_true',help='for every sv type add the other types to the NOT.1.0 background label\t[False]')
    parser.add_argument('--data_split',type=float,help='test to validation ration\t[5.0/6.0]')
    parser.add_argument('--split_seed', type=int, help='random splitting seed\t[None]')
    parser.add_argument('--sub_sample',type=float,help='proportion of sv labels to sample\t[1.0]')
    parser.add_argument('--sub_sample_not',type=float,help='proportion of background labels to sample (overriden by balance)\t[0.5]')
    parser.add_argument('--sample_batch',type=int,help='how many sample to use at each generator pass (overriden by data_split)\t[16]')
    parser.add_argument('--mem_sm_limit',type=int,help='sample switch point for OOC version of training\t[32]')
    parser.add_argument('--gpu_num',type=int,help='pick one of your available logical gpus\t[None]')
    parser.add_argument('--mmnt_idx',type=str,help='comma-seperated list of moment indecies, where 0 is moment 1, etc...\t[0,1,2,3]')
    parser.add_argument('--tracks',type=str,help='comma-seperated list of tracks to use for training/testing\t[all]')
    parser.add_argument('--form',type=str,help='cnn,r_lstm_cnn, br_lstm, r_gru_cnn, br_gru_cnn\t[cnn]')
    parser.add_argument('--filters',type=str,help='comma-seperated list of filters\t[sml,med,lrg,all]')
    parser.add_argument('--sub_filters',type=str,help='comma-seperated list of sub filters\t[L,LR,R]')
    parser.add_argument('--levels', type=str, help='comma-seperated list of NN level depth\t[2]')
    parser.add_argument('--cmxs',type=str,help='comma-seperated list of NN complexity\t[4]')
    parser.add_argument('--batches',type=str,help='comma-seperated list of training batch size\t[32]')
    parser.add_argument('--epochs',type=str,help='comma-seperated list of training epochs\t[10]')
    parser.add_argument('--decays',type=str, help='comma-seperated sci-notation aware training L2-decay rates\t[1e-5]')
    parser.add_argument('--verbose',action='store_true',help='output messages at each step\t[False]')
    args = parser.parse_args()
    if args.in_path is not None:        data = args.in_path
    else: raise IOError
    if args.out_path is not None:       out_dir = args.out_path
    else:                               out_dir = data+'/training_runs/'
    if not os.path.exists(out_dir):     os.mkdir(out_dir)
    if args.geno_bins is not None:      geno_bins = sorted([float(x) for x in args.geno_bins.rsplit(',')])
    else:                               geno_bins = [0.5,1.0]
    if args.sv_types is not None:       svs    = args.sv_types.rsplit(',')
    else:                               svs    = ['DEL']
    if args.no_geno:                    labels = {sv:{'NOT':0,'%s'%sv:1} for sv in svs}
    else:
        labels = {sv:{} for sv in svs}
        for sv in labels:
            labels[sv] = {'NOT.1.0':0}
        for sv in labels:
            i = 1
            for g in geno_bins:
                labels[sv]['%s.%s'%(sv,g)] = i
                i += 1
    if args.geno_cut is not None:       geno_cut = args.geno_cut
    else:                               geno_cut = 0.0
    if args.balance:                    balance_not = True
    else:                               balance_not = False
    if args.short_test:                 short_test  = True
    else:                               short_test  = False
    if args.data_split is not None:     data_split = args.data_split
    else:                               data_split = 5.0/6.0
    if args.split_seed is not None:     split_seed = args.split_seed
    else:                               split_seed = np.random.get_state()[1][0]
    if args.sub_sample is not None:     sub_sample = args.sub_sample
    else:                               sub_sample = 1.0
    if args.sub_sample_not is not None: sub_sample_not = args.sub_sample_not
    else:                               sub_sample_not = 1.0
    if args.sample_batch is not None:   sample_batch = args.sample_batch
    else:                               sample_batch = 32
    if args.mem_sm_limit is not None:   mem_sm_limit = args.mem_sm_limit
    else:                               mem_sm_limit = 32
    if args.gpu_num is not None:        gpu_num  = args.gpu_num
    else:                               gpu_num  = 0
    if args.mmnt_idx is not None:       mmnt_idx = [int(x) for x in args.mmnt_idx.split(',')]
    else:                               mmnt_idx = [0,1,2,3]
    if args.tracks is not None:         tracks   = args.tracks.split(',')
    else:                               tracks   = 'all'
    if args.filters is not None:        filters  = args.filters.rsplit(',')
    else:                               filters  = ['sml','med','lrg','all']
    if args.sub_filters is not None:    sub_filters  = args.sub_filters.rsplit(',')
    else:                               sub_filters  = ['L','M','R','LR']
    if args.levels is not None:         levels   = [int(l) for l in args.levels.rsplit(',')]
    else:                               levels   = [2]
    if args.cmxs is not None:           cmxs     = [int(cmx) for cmx in args.cmxs.rsplit(',')]
    else:                               cmxs     = [4]
    if args.batches is not None:        batches  = [int(batch) for batch in args.batches.rsplit(',')]
    else:                               batches  = [32]
    if args.epochs is not None:         epochs   = [int(epoch) for epoch in args.epochs.rsplit(',')]
    else:                               epochs   = [10]
    if args.decays is not None:         decays   = [float(decay) for decay in args.decays.rsplit(',')]
    else:                               decays   = [1e-5]
    if args.form is not None:           form     = args.form
    else:                               form     = 'cnn'
    if args.verbose:                    verbose  = True
    else:                               verbose  = False
    #-------------------------------------------------------------------------------------------
    sv = svs[0] #labels are the same for sv types
    labels_idx = {labels[sv][l]:l for l in labels[sv]}
    classes    = len(labels[sv])
    print('using data labels: %s'%labels)
    #------------------------------------INPUT SAMPLES AND PARTIONING------------------------------------------------------------
    f = File(data,'r') #would need to iterate over all three inputs
    total_sample_num = len(list(f.keys()))
    print('found %s total samples in the input data tensor set'%total_sample_num)
    np.random.seed(split_seed)
    print('using split_seed=%s'%split_seed)
    test_sms  = sorted(list(np.random.choice(list(f.keys()),int(round((1.0-data_split)*len(f.keys()))),replace=False)))
    work_sms  = list(set(f.keys()).difference(set(test_sms)))
    train_sms = sorted(list(np.random.choice(work_sms,len(f.keys())-int(round(2*(1.0-data_split)*len(f.keys()))),replace=False)))
    valid_sms = sorted(list(set(work_sms).difference(set(train_sms))))
    print('train_sms=%s'%','.join(train_sms))
    print('valid_sms=%s' % ','.join(valid_sms))
    print('test_sms=%s' % ','.join(test_sms))
    sm = list(f.keys())[0] #get the dimensional attributed for the modeling....
    trks,mmnts = [],[]
    dims,w,n = None,None,None   #[windows,features,moments]
    k     = list(f[sm].keys())[0]
    trks  = f[sm][k].attrs['feature_order']
    trk_idx = {trks[i]:i for i in range(len(trks))}
    trk_xdi = {i:trks[i] for i in range(len(trks))}
    mmnts = [x for x in range(f[sm][k].shape[3])]
    if len(f[sm][k])>0:
        dims = f[sm][k][0].shape
        n = int(dims[0]//5) #5-point topology generator
        w = int(f[sm][k].attrs['w'])
    if len(trks)>0 and len(mmnts)>0:
        print('tracks available are: %s'%trks)
        if tracks=='all': tracks = range(len(trks)) #all the indecies of the data
        else:             tracks = sorted([trk_idx[trk] for trk in set(tracks).intersection(set(trks))])
        moments = sorted(set(mmnt_idx).intersection(set(mmnts)))
        dims = (dims[0],len(tracks),len(moments))
    else: raise AttributeError
    if dims is None or w is None or n is None: raise AttributeError #now we have the dimensional information
    f.close()
    #----------------------------------------------------------------------------------------------------------------------------
    params,F,best_run = [],{},{}
    train_cache,valid_cache,test_cache = None,None,None
    with tf.device('/gpu:%s'%gpu_num):
        for sv in svs:
            F[sv] = {}
            if 'sml' in filters:
                if 'LR' in sub_filters:
                    F[sv]['sml-%s-LR'%sv] = {'idx':[0,5*n],  'size':[0,w*n],'tracks':tracks,'moments':moments}
            if 'med' in filters:
                if 'LR' in sub_filters:
                    F[sv]['med-%s-LR'%sv] = {'idx':[0,5*n],  'size':[w*n,2*w*n],'tracks':tracks,'moments':moments}
                if 'L' in sub_filters:
                    F[sv]['ml-%s-L'%sv]   = {'idx':[0,3*n],  'size':[w*n,int(1E9)],'tracks':tracks,'moments':moments}
                if 'R' in sub_filters:
                    F[sv]['ml-%s-R'%sv]   = {'idx':[2*n,5*n],'size':[w*n,int(1E9)],'tracks':tracks,'moments':moments}
            if 'lrg' in filters:
                if 'LR' in sub_filters:
                    F[sv]['ml-%s-LR'%sv]   = {'idx':[0,3*n],  'size':[w*n,int(1E9)],'tracks':tracks,'moments':moments}
                if 'R' in sub_filters:
                    F[sv]['ml-%s-R'%sv]   = {'idx':[2*n,5*n],'size':[w*n,int(1E9)],'tracks':tracks,'moments':moments}
                if 'M' in sub_filters:
                    F[sv]['lrg-%s-M'%sv]  = {'idx':[2*n-n//2,3*n+n//2],'size':[2*w*n,int(1E9)],'tracks':tracks,'moments':moments}
                if 'LR' in sub_filters:
                    F[sv]['lrg-%s-LR'%sv] = {'idx':[0,5*n],  'size':[2*w*n,int(1E9)],'tracks':tracks,'moments':moments}
            if 'all' in filters:
                if 'L' in sub_filters:
                    F[sv]['all-%s-L'%sv]  = {'idx':[0,2*n],  'size':[0,int(1E9)],'tracks':tracks,'moments':moments}
                if 'R' in sub_filters:
                    F[sv]['all-%s-R'%sv]  = {'idx':[3*n,5*n],'size':[0,int(1E9)],'tracks':tracks,'moments':moments}
                if 'LR' in sub_filters:
                    F[sv]['all-%s-LR'%sv] = {'idx':[0,5*n],  'size':[0,int(1E9)],'tracks':tracks,'moments':moments}
            if 'zoom' in filters:
                if 'LR' in sub_filters:
                    F[sv]['L0-%s-LR'%sv] = {'idx':[0,5*n],  'size':[int(0),  int(1E6)],'tracks':tracks,'moments':moments}
                # F[sv]['L1-%s-LR'%sv] = {'idx':[0,5*n],  'size':[int(0),  int(1E2)],'tracks':tracks,'moments':moments}
                # F[sv]['L2-%s-LR'%sv] = {'idx':[0,5*n],  'size':[int(1E2),int(1E3)],'tracks':tracks,'moments':moments}
                # F[sv]['L3-%s-LR'%sv] = {'idx':[0,5*n],  'size':[int(1E3),int(1E4)],'tracks':tracks,'moments':moments}
                # F[sv]['L4-%s-LR'%sv] = {'idx':[0,5*n],  'size':[int(1E4),int(1E6)],'tracks':tracks,'moments':moments}
            for l in levels:
                for c in cmxs:
                    for b in batches:
                        for e in epochs:
                            for d in decays:
                                params += [[sv,l,c,b,e,d]]
        for sv in F: # DEL:{'sml-DEL-LR':{'idx':[0,55],'size':[0,w*n],...}
            for filt in F[sv]:
                run_score_path = out_dir+'/scores/%s.score.json'%filt
                if os.path.exists(run_score_path):
                    print('found prior run file %s, setting it up...'%run_score_path)
                    with open(run_score_path,'r') as f:
                        best_run[filt] = json.load(f)
                else:
                    best_run[filt] = {'score':0.0,'long_score':0.0,'params':{}}
        # raise IOError
        #-------------------------------------------------------------------------------------------
        start,run_num = time.time(),0
        M,S = {},{}
        for param in params:
            sv,level,cmx,batch,epoch,decay = param
            run_num += 1
            print('\n:::::::::::::::::::::::::::::::::::::::: starting training run %s/%s ::::::::::::::::::::::::::::::::::::::::::::'%(run_num,len(params)))
            for filt in F[sv]:
                long_score = 0.0
                print('lvls=%s,cmx=%s,batch=%s,epoch=%s,decay=%s'%(level,cmx,batch,epoch,decay))
                M = {'score':{},'hist':{}}
                S = {'params':{'labels':labels[sv],'idx':F[sv][filt]['idx'],'size':F[sv][filt]['size'],'w':w,'n':n,
                               'cmx':cmx,'levels':level,'batch':batch,'epoch':epoch,'decay':decay,'balance':balance_not}}
                f_start = time.time()
                print('_____________________________________________________________________________')
                print('training and testing filter F=%s, size_range=[%s,%s], tracks=%s, moments=%s'%\
                      (filt,F[sv][filt]['size'][0],F[sv][filt]['size'][1],[trk_xdi[x] for x in F[sv][filt]['tracks']],F[sv][filt]['moments']))
                #load train/test/validate? data-----------------------------------------------------------------------------------------
                if total_sample_num>mem_sm_limit:
                    train_gen = DataGen(data_path=data,sms_list=train_sms,labels=labels[sv],size_range=F[sv][filt]['size'],
                                        sub_sample=sub_sample,sub_sample_not=sub_sample_not,balance_not=balance_not,
                                        cols=F[sv][filt]['idx'],ftrs=F[sv][filt]['tracks'],mmnts=F[sv][filt]['moments'],
                                        sample_batch=min(sample_batch,len(train_sms)),batch=batch,verbose=verbose)
                    valid_gen = DataGen(data_path=data,sms_list=valid_sms,labels=labels[sv],size_range=F[sv][filt]['size'],
                                        sub_sample=sub_sample,sub_sample_not=sub_sample_not,balance_not=balance_not,
                                        cols=F[sv][filt]['idx'],ftrs=F[sv][filt]['tracks'],mmnts=F[sv][filt]['moments'],
                                        sample_batch=min(sample_batch,len(valid_sms)),batch=batch,verbose=False)
                    test_gen  = data_gen(data_path=data,sms_list=test_sms,labels=labels[sv],size_range=F[sv][filt]['size'],
                                         sub_sample=sub_sample,sub_sample_not=sub_sample_not,balance_not=balance_not,
                                         cols=F[sv][filt]['idx'],ftrs=F[sv][filt]['tracks'],mmnts=F[sv][filt]['moments'],
                                         sample_batch=min(sample_batch,len(test_sms)),batch=1024,verbose=False)
                    #build up a model<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    model = build_model(dims=(F[sv][filt]['idx'][1]-F[sv][filt]['idx'][0],)+dims[1:],
                                        classes=classes,levels=level,cmx=cmx,decay=decay,form=form)
                    #train it::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                    #if verbose: model.summary()
                    m_start = time.time()
                    print('fitting filter=%s with %s training samples against %s validation samples'%(filt,len(train_sms),len(valid_sms)))
                    if train_gen.X.shape[0]>0:
                        model.fit_generator(train_gen,
                                            epochs=epoch,
                                            validation_data=valid_gen,
                                            verbose=False,workers=4)
                        m_stop = time.time()
                        print('filter=%s model fit time was %s sec'%(filt,round(m_stop-m_start,2)))
                        #test it------------------------------------------------------------------#will test against unseen samples
                        test_start = time.time()
                        print('starting evaluation on %s reserved non-training/validation (test) samples'%len(test_sms))
                        prs,lbls = [],[]
                        for i in range(min(sample_batch,len(test_sms))):
                            (xs,ys) = next(test_gen)
                            prs  += [model.predict(xs)]
                            lbls += [ys]
                            if verbose: print('%s predictions made for evaluating filt=%s'%(len(xs),filt))
                        pred,lbls = np.concatenate(prs),np.concatenate(lbls)
                        test_stop = time.time()
                        if verbose: print('finished prediction in % sec, now scoring %s test samples'%(round(test_stop-test_start,2),len(test_sms)))
                        M['hist']  = model.history.history
                        M['score'] = confusion_matrix([np.argmax(x) for x in pred],[y[0] for y in lbls],classes)
                        prec,rec,f1 = metrics(M['score'])
                        print('F=%s prec =%s'%(filt,{k:round(prec[k],4) for k in sorted(prec)}))
                        print('F=%s rec  =%s'%(filt,{k:round(rec[k],4)  for k in sorted(rec) }))
                        print('F=%s f1   =%s'%(filt,{k:round(f1[k],4)   for k in sorted(f1)  }))
                        run_score = sum([f1[k] for k in f1])/3.0
                        print('averaged f1 score for classes=%s : %s'%([labels_idx[k] for k in f1.keys()],round(run_score,5)))
                        S['run_score'] = run_score
                        print('-----------------------------------------------------------------------------')
                        if run_score>best_run[filt]['score']: #scored better on unseen data than other runs...
                            print('*** best_run score for filt=%s increased from %s to %s ***'%(filt,round(best_run[filt]['score'],5),round(run_score,5)))
                            model_path = out_dir+'/models/'
                            if not os.path.exists(model_path): os.mkdir(model_path)
                            model.save(model_path+'/%s.model.hdf5'%(filt))
                            plot_model(model, model_path)
                            plot_path = out_dir+'/plots/'
                            if not os.path.exists(plot_path): os.mkdir(plot_path)
                            score_path = out_dir+'/scores/'
                            title = 'lvls=%s cmx=%s batch=%s decay=%s balance=%s %s score'
                            plot_confusion_heatmap(M['score'],title%(level,cmx,batch,decay,balance_not,filt),out_path=plot_path+'/%s.score.jpg'%(filt))
                            title = 'lvls=%s cmx=%s batch=%s decay=%s balance=%s %s modeling'
                            plot_train_test(M['hist'],title%(level,cmx,batch,decay,balance_not,filt),out_path=plot_path+'/%s.model.jpg'%(filt))
                            if not os.path.exists(score_path): os.mkdir(score_path)
                            best_run[filt]['score'],best_run[filt]['params']  = S['run_score'],S['params']
                            with open(score_path+'%s.score.json'%filt,'w') as f: json.dump(best_run[filt],f)
                        f_stop = time.time()
                        if verbose: print('filter %s training time was %s'%(filt,round(f_stop-f_start,2)))
                    else:
                        print('skipping filt=%s no data!'%filt)
                else:
                    msg = '/\/\ loading data into memory for %s training samples against %s validation samples for filt=%s \/\/'
                    print(msg%(len(train_sms),len(valid_sms),filt))
                    load_start = time.time()
                    if train_cache is None: train_cache = data_cache(data,sms=train_sms)
                    else: print('training data was cached, moving on...')
                    # raise IOError
                    train_data = data_load(data_in=train_cache,sms_list=train_sms,labels=labels[sv],size_range=F[sv][filt]['size'],
                                           sub_sample=sub_sample,sub_sample_not=sub_sample_not,balance_not=balance_not,counter_label=args.counter_label,
                                           cols=F[sv][filt]['idx'],ftrs=F[sv][filt]['tracks'],mmnts=F[sv][filt]['moments'],
                                           geno_cut=geno_cut,sample_batch=min(sample_batch,len(train_sms)),verbose=verbose)
                    if valid_cache is None: valid_cache = data_cache(data,sms=valid_sms)
                    else: print('validation data was cached, moving on...')
                    valid_data = data_load(data_in=valid_cache,sms_list=valid_sms,labels=labels[sv],size_range=F[sv][filt]['size'],
                                           sub_sample=sub_sample,sub_sample_not=sub_sample_not,balance_not=balance_not,counter_label=args.counter_label,
                                           cols=F[sv][filt]['idx'],ftrs=F[sv][filt]['tracks'],mmnts=F[sv][filt]['moments'],
                                           geno_cut=geno_cut,sample_batch=min(sample_batch,len(valid_sms)),verbose=False)
                    if test_cache is None: test_cache = data_cache(data,sms=test_sms)
                    else: print('testing data was cached, moving on...')
                    test_data  = data_load(data_in=test_cache,sms_list=test_sms,labels=labels[sv],size_range=F[sv][filt]['size'],
                                           sub_sample=sub_sample,sub_sample_not=sub_sample_not,balance_not=balance_not,counter_label=args.counter_label,
                                           cols=F[sv][filt]['idx'],ftrs=F[sv][filt]['tracks'],mmnts=F[sv][filt]['moments'],
                                           geno_cut=geno_cut,sample_batch=min(sample_batch,len(test_sms)),verbose=False)
                    load_stop = time.time()
                    print('total data loading in %s sec'%round(load_stop-load_start,2))
                    print('----------------------------------------------------------------------------------')
                    #summary of total instances in the pool?
                    for c in range(classes):
                        print('%s total training instances of class=%s'%(len(np.where(train_data[1]==c)[0]),{labels[sv][l]:l for l in labels[sv]}[c]))
                    for c in range(classes):
                        print('%s total testing instances of class=%s'%(len(np.where(test_data[1]==c)[0]),{labels[sv][l]:l for l in labels[sv]}[c]))
                    print('----------------------------------------------------------------------------------')
                    #build up a model<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    model = build_model(dims=(F[sv][filt]['idx'][1]-F[sv][filt]['idx'][0],)+dims[1:],
                                        classes=classes,levels=level,cmx=cmx,decay=decay,form=form)
                    #train it::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                    m_start = time.time()
                    print('fitting filter=%s with %s training samples against %s validation samples'%(filt,len(train_sms),len(valid_sms)))
                    if len(train_data[0])>0 and len(valid_data[0])>0 and len(test_data[0])>0:
                        print('fitting training data into CPU/GPU memory')
                        model_path = out_dir+'/models/'
                        if not os.path.exists(model_path): os.mkdir(model_path)
                        if not os.path.exists(model_path+'/temp/'): os.mkdir(model_path+'/temp/')
                        model_check = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'/temp/%s.model.hdf5'%filt,
                                                                         monitor='sparse_categorical_accuracy',
                                                                         save_best_only=True,save_freq='epoch')
                        model_stop = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy',patience=10)
                        model.fit(x=train_data[0],y=train_data[1],
                                  batch_size=batch,
                                  epochs=epoch,
                                  validation_data=(valid_data[0],valid_data[1]),
                                  shuffle=True,
                                  callbacks=[model_check,model_stop],
                                  verbose=False)
                        m_stop = time.time()
                        print('filter=%s model fit time was %s sec'%(filt,round(m_stop-m_start,2)))

                        #test it------------------------------------------------------------------#will test against unseen samples
                        test_start = time.time()
                        print('starting evaluation on %s reserved non-training/validation (test) samples'%len(test_sms))
                        pred,lbls = model.predict(test_data[0]),test_data[1]
                        test_stop = time.time()
                        if verbose: print('finished prediction in % sec, now scoring %s test samples'%(round(test_stop-test_start,2),len(test_sms)))
                        M['hist']  = model.history.history
                        M['score'] = confusion_matrix([np.argmax(x) for x in pred],[y[0] for y in lbls],classes)
                        prec,rec,f1 = metrics(M['score'])
                        print('F=%s prec =%s'%(filt,{k:round(prec[k],4) for k in sorted(prec)}))
                        print('F=%s rec  =%s'%(filt,{k:round(rec[k],4)  for k in sorted(rec) }))
                        print('F=%s f1   =%s'%(filt,{k:round(f1[k],4)   for k in sorted(f1)  }))
                        run_score = sum([f1[k] for k in f1])/classes
                        print('averaged f1 score for classes=%s : %s'%([labels_idx[k] for k in f1.keys()],round(run_score,5)))
                        S['run_score'] = run_score

                        if not short_test: #do the longer full sequence testing...
                            test_start = time.time()
                            score_path = out_dir + '/test/scores/'
                            if not os.path.exists(out_dir+'/test/'): os.mkdir(out_dir+'/test/')
                            if not os.path.exists(score_path): os.mkdir(score_path)
                            filt_run = {'score':S['run_score'],'long_score':long_score,
                                        'params':S['params'],'metrics':metrics(M['score'])}
                            with open(score_path+'%s.score.json'%filt,'w') as f: json.dump(filt_run,f)
                            if not os.path.exists(out_dir+'/test/models/'): os.mkdir(out_dir+'/test/models/')
                            models_present = [os.path.exists(out_dir+'/models/%s.model.hdf5'%flt) for flt in F[sv]]
                            model.save(out_dir+'/test/models/%s.model.hdf5'%(filt))
                            if all(models_present):
                                for flt in set(F[sv]).difference(set([filt])):
                                    shutil.copyfile(out_dir+'/models/%s.model.hdf5'%flt,out_dir+'/test/models/%s.model.hdf5'%flt)
                                base_dir = '/'.join(args.in_path.rsplit('/')[:-2])+'/'
                                #[0] get test_sms from filenames
                                sms_hdf5 = glob.glob(base_dir+'/tensors/*.norm.hdf5')
                                test_sms_hdf5 = {}
                                for sm in test_sms:
                                    for sm_path in sms_hdf5:
                                        if sm_path.find(sm)>=0: test_sms_hdf5[sm_path.rsplit('/')[-1].rsplit('.')[0]] = sm_path
                                sm = sorted(test_sms)[0]
                                f_seqs = File(test_sms_hdf5[sorted(test_sms_hdf5)[0]],'r')
                                available_seqs = sorted(f_seqs[sorted(f_seqs)[0]]['all'],key=lambda x: x.zfill(255))
                                f_seqs.close()
                                select_seqs = available_seqs[-1*len(available_seqs)//5+1:] #1/5 of sequences
                                svmachina_path = os.path.dirname(os.path.realpath(__file__))+'/'
                                command    = ['python3',svmachina_path+'predict_sv.py','--samples %s'%','.join(sorted(test_sms_hdf5)),
                                              '--sv_type %s'%sv,'--seqs %s'%','.join(select_seqs),
                                              '--base_dir %s'%base_dir,'--run_dir %s'%out_dir,
                                              '--out_dir %s/test/'%out_dir,'--gpu %s'%gpu_num]
                                out = ''
                                try:
                                    print(' '.join(command))
                                    out += subprocess.check_output(' '.join(command),shell=True).decode('UTF-8')
                                except Exception as E:
                                    print(E)
                                    pass
                                print(out)
                                for sm in test_sms:
                                    if os.path.exists(out_dir+'/test/%s.%s.tensorsv.pickle.gz'%(sm,sv)):
                                        with gzip.GzipFile(out_dir+'/test/%s.%s.tensorsv.pickle.gz'%(sm,sv),'rb') as f:
                                            D = pickle.load(f)
                                    long_score += D[sm][sv]['opt_score']
                                long_score /= len(test_sms)
                            test_stop  = time.time()
                        print('-----------------------------------------------------------------------------')
                        if run_score>best_run[filt]['score'] and long_score<=0.0: #scored better on unseen data than other runs...
                            print('*** best_run score for filt=%s increased from %s to %s ***'%(filt,round(best_run[filt]['score'],5),round(run_score,5)))
                            model.save(model_path+'/%s.model.hdf5'%(filt))
                            plot_model(model, model_path)
                            plot_path = out_dir+'/plots/'
                            if not os.path.exists(plot_path): os.mkdir(plot_path)
                            score_path = out_dir+'/scores/'
                            title = 'lvls=%s cmx=%s batch=%s decay=%s balance=%s %s score'
                            plot_confusion_heatmap(M['score'],title%(level,cmx,batch,decay,balance_not,filt),out_path=plot_path+'/%s.score.jpg'%(filt))
                            title = 'lvls=%s cmx=%s batch=%s decay=%s balance=%s %s modeling'
                            plot_train_test(M['hist'],title%(level,cmx,batch,decay,balance_not,filt),out_path=plot_path+'/%s.model.jpg'%(filt))
                            if not os.path.exists(score_path): os.mkdir(score_path)
                            best_run[filt]['score'],best_run[filt]['params']  = S['run_score'],S['params']
                            best_run[filt]['long_score'] = long_score
                            best_run[filt]['metrics'] = metrics(M['score'])
                            with open(score_path+'%s.score.json'%filt,'w') as f: json.dump(best_run[filt],f)
                        if long_score>best_run[filt]['long_score']:
                            print('*** best_run long_score for filt=%s increased from %s to %s ***'%\
                                  (filt,round(best_run[filt]['long_score'],5),round(long_score,5)))
                            model.save(model_path+'/%s.model.hdf5'%(filt))
                            plot_model(model, model_path)
                            plot_path = out_dir+'/plots/'
                            if not os.path.exists(plot_path): os.mkdir(plot_path)
                            score_path = out_dir+'/scores/'
                            title = 'lvls=%s cmx=%s batch=%s decay=%s balance=%s %s score'
                            plot_confusion_heatmap(M['score'],title%(level,cmx,batch,decay,balance_not,filt),out_path=plot_path+'/%s.score.jpg'%(filt))
                            title = 'lvls=%s cmx=%s batch=%s decay=%s balance=%s %s modeling'
                            plot_train_test(M['hist'],title%(level,cmx,batch,decay,balance_not,filt),out_path=plot_path+'/%s.model.jpg'%(filt))
                            if not os.path.exists(score_path): os.mkdir(score_path)
                            best_run[filt]['score'],best_run[filt]['params']  = S['run_score'],S['params']
                            best_run[filt]['long_score'] = long_score
                            best_run[filt]['metrics'] = metrics(M['score'])
                            with open(score_path+'%s.score.json'%filt,'w') as f: json.dump(best_run[filt],f)
                        f_stop = time.time()
                        if verbose: print('filter %s training time was %s'%(filt,round(f_stop-f_start,2)))
        stop = time.time()
        print('total training time for the %s models and params: %s was %s sec'%(sorted(list(F.keys())),params,round(stop-start,2)))