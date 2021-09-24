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
import hashlib
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

#!/usr/env/bin/python3
import os
import gzip
import glob
import pickle

def geno_split_sniff(vcf_path,skip='#',delim='\t',info_i=7):
    print('sniffing for genotype splitter format in VCF file...')
    s,g,h = '',[],[]
    if vcf_path.endswith('.gz'):
        with gzip.GzipFile(vcf_path,'rb') as f:
            s = ''.join([l.decode() for l in f.readlines()])
    elif vcf_path.endswith('.vcf'):
        with open(vcf_path,'r') as f:
            s = ''.join(f.readlines())
    else:
        print('incorrect vcf file suffix...')
    while s[-1]=='\n': s = s[:-1] #clean up any loose '\n'
    for t in s.split('\n'):
        if len(g)>10: break
        if not t.startswith(skip): g += [t.split(delim)]
        else:                      h += [t]
    if h[-1].split(delim)[-1]=='INFO': splitter = None
    else:
        if g[0][info_i+3].find('/')>=0: splitter = '/'
        if g[0][info_i+3].find('|')>=0: splitter = '|'
    return splitter

def geno_vcf_regions_to_dict(vcf_path,skip='#',delim='\t',info_i=7):
    S = {}
    s,g,h = '',[],[]
    if vcf_path.endswith('.gz'):
        with gzip.GzipFile(vcf_path,'rb') as f:
            s = ''.join([l.decode() for l in f.readlines()])
    elif vcf_path.endswith('.vcf'):
        with open(vcf_path,'r') as f:
            s = ''.join(f.readlines())
    else:
        print('incorrect vcf file suffix...')
    while s[-1]=='\n': s = s[:-1] #clean up any loose '\n'
    for t in s.split('\n'):
        if not t.startswith(skip): g += [t.split(delim)]
        else:                      h += [t]
    for row in g:
        sv = row[info_i].split('SVTYPE=')[-1].split(';')[0]
        seq,start,end =row[0],int(row[1]),int(row[info_i].split(';END=')[-1].split(';')[0])
        if sv in S:
            if seq in S[sv]: S[sv][seq] += [[seq,start,end,max(1,end-start+1)]]
            else:            S[sv][seq]  = [[seq,start,end,max(1,end-start+1)]]
        else:                S[sv] =  {seq:[[seq,start,end,max(1,end-start+1)]]}
    for sv in S:
        for seq in S[sv]:
            S[sv][seq] = sorted(S[sv][seq],key=lambda x: (x[0].zfill(255),x[1]))
    return S

#P = {sv:{seq:[vc1,vc2,etc]}}, vcf_in is the orginal VCF file to use, vcf_out is the output VCF file
# coord_match is used when you wish to utilize the model scanning proceedure to update breakpoints
def write_vcf(P,models,sm,vcf_in,vcf_out,coord_match=False,skip='#',delim='\t',info_i=7,gz=True):
    M = {}
    for sv in models: M[sv] = get_model_hash(models[sv])
    s,g,h = '',[],[]
    if vcf_in.endswith('.gz'):
        with gzip.GzipFile(vcf_in,'rb') as f:
            s = ''.join([l.decode() for l in f.readlines()])
    elif vcf_in.endswith('.vcf'):
        with open(vcf_in,'r') as f:
            s = ''.join(f.readlines())
    else:
        print('incorrect vcf file suffix...')
    while s[-1]=='\n': s = s[:-1] #clean up any loose '\n'
    for t in s.split('\n'):
        if not t.startswith(skip): g += [t.split(delim)]
        else:                      h += [t]
    h[-1] += '\tFORMAT\t%s'%sm
    model_info = ['##INFO=<ID=MODEL_CLASS,Number=1,Type=String,Description="Maximal Filtered TensorSV Predicted Model Class">',
                  '##INFO=<ID=MODEL_LIKELIHOOD,Number=1,Type=Float,Description="TensorSV Predicted Model Class Likelihood Under the Model">',
                  '##INFO=<ID=MODEL_MD5,Number=1,Type=String,Description="MD5 Hash of the TensorSV Predictive Model">']
    h = h[:-1]+model_info+[h[-1]]
    G = {}
    for row in g:
        sv = row[info_i].split('SVTYPE=')[-1].split(';')[0]
        seq,start,end =row[0],int(row[1]),int(row[info_i].split(';END=')[-1].split(';')[0])
        k = (sv,seq,start,end)
        if k not in G: G[k] = row
    if len(G)==len(g): print('all SVs uniqiue from input vcf file=%s'%vcf_in)
    else:              print('%s unique SVs out of %s total SVs from input vcf file=%s'%(len(G),len(g),vcf_in))
    if coord_match: print('not implemented yet...')
    else:
        V = {}
        for sv in P:
            for seq in P[sv]:
                for vc in P[sv][seq]:
                    k = (sv,vc[0],vc[1],vc[2])
                    if k not in G: print('error with vc key=%s in G'%k)
                    else:
                        V[k] = G[k]  #need to add the genotype information and the model likelihood, model hash
                        if V[k][info_i].find('SVLEN=')<0: V[k][info_i] += ';SVLEN=%s'%(vc[2]-vc[1]+1)
                        V[k][info_i] += ';MODEL_CLASS=%s.%s;MODEL_LIKELIHOOD=%s;MODEL_MD5=%s'%(sv,vc[4],vc[5],M[sv])
                        V[k] += ['GT',('1/1' if vc[4]==1.0 else '0/1')]
    #write it out now...
    s = '\n'.join(h)+'\n'
    for k in sorted(V,key=lambda x: (x[1].zfill(255),x[2])):
        s += '\t'.join(V[k])+'\n'
    if gz:
        with gzip.GzipFile(vcf_out+'/%s.tensorsv.geno.vcf.gz'%sm,'wb') as f:
            f.write(s.encode('UTF-8'))
    else:
        with open(vcf_out+'/%s.tensorsv.geno.vcf'%sm,'w') as f:
            f.write(s)
    print('finished writing VCF to disk')
    return True

def svtyper_sample_vcf_to_dict(vcf_path,skip='#',delim='\t',alt_i=4,info_i=7,s_i=9,geno_split='/'):
    print('generating new dict pickle from vcf input...')
    s,g,h = '',[],[]
    if vcf_path.endswith('.gz'):
        with gzip.GzipFile(vcf_path,'rb') as f:
            s = ''.join([l.decode() for l in f.readlines()])
    elif vcf_path.endswith('.vcf'):
        with open(vcf_path,'r') as f:
            s = ''.join(f.readlines())
    else:
        print('incorrect vcf file suffix...')
    while s[-1]=='\n': s = s[:-1] #clean up any loose '\n'
    for t in s.split('\n'):
        if not t.startswith(skip): g += [t.split(delim)]
        else:                      h += [t]
    sample_ids = h[-1].split(delim)[s_i:] #2504 sample names/ids
    sdx_i = {sample_ids[i]:i for i in range(len(sample_ids))}
    samples = sample_ids
    S,T = {s:{} for s in samples},{}
    row_num = 0
    for row in g:
        start = int(row[1])
        t     = row[info_i].split('SVTYPE=')[-1].split(';')[0]
        if row[alt_i].find('INS')>0:
            t,end = 'INS',start
        else:
            if row[info_i].find('END=')>=0:
                end = int(row[info_i].split(';END=')[1].split(';')[0])
            else:
                end = start+1
        svl   = abs(end-start)
        af = '1.0'
        if row[alt_i].find('CN')>0: #each sample only has one of the possible alleles!
            cns  = [int(x.replace('<','').replace('>','').replace('CN','')) for x in row[alt_i].split(',')]
            af   = [float(x) for x in af.split(',')]
        else:
            cns = [2 for x in row[alt_i].split(',')]
            af  = float(af)
        lt = t
        a_id  = row[2]
        for s in sdx_i: #unpack some of the VCF file format into a dict
            gn = row[s_i:][sdx_i[s]]  #get genotype information
            ls = len(gn.split(geno_split))
            if gn=='.' or gn.startswith('./.'):
                gn = geno_split.join(['0' for x in range(ls)])   #check for missing values='.'
            gh = [int(x) for x in gn.split(':')[0].split(geno_split)]    #split for genotype parsing
            gt = sum(gh)                                               #sum > 0 => variation present
            if gt>0 and s in samples: #only processes the g1kp3 samples you need: #genotype is present: 1|0,0|1,1|1................................
                ap = 0
                for i in range(len(gh)):
                    if gh[i]>0: ap = i
                ht,cn = 1.0*sum([1 if x==gh[ap] else 0 for x in gh])/ls,ls
                if t=='DEL':
                    cn = sum([1 if x==0 else 0 for x in gh]) # cn = 0 or 1 for ploidy 1-2
                    naf = af
                elif t=='DUP' or t=='CNV': #calculate het and cn for DEL/DUP/CNV dynamic
                    cn,dels,gns = 0,0,[1]+cns
                    if type(af) is float: naf = [af]
                    else:                 naf = [1.0-sum(af)]+af
                    maf = 0.0
                    for i in range(len(gh)):
                        if gns[gh[i]]==0: dels += 1
                        if gns[gh[i]] != 1:
                            if gh[i]<len(naf)-1: maf = max(maf,naf[gh[i]])
                        cn  += gns[gh[i]]
                    naf = maf
                    if cn<2:              t = 'DEL'
                    elif cn==2:           t = 'CNV'
                    elif cn>2 and dels<1: t = 'DUP'
                    else:                 t = 'CNV'
                else: naf = af
                if t in S[s]: S[s][t] += [[row[0],start,end,svl,ht,naf,cn,a_id]]
                else:         S[s][t]  = [[row[0],start,end,svl,ht,naf,cn,a_id]]
                if t in T:
                    if a_id in T[t]: T[t][a_id][s] = [row[0],start,end,svl,ht,naf,cn,a_id]
                    else:            T[t][a_id] = {s:[row[0],start,end,svl,ht,naf,cn,a_id]}
                else:                T[t] = {a_id:{s:[row[0],start,end,svl,ht,naf,cn,a_id]}}
            t = lt
        row_num += 1
        if row_num%1000==0: print('%s rows processed'%row_num)
    for s in S:
        for t in S[s]:
            S[s][t] = sorted(S[s][t],key= lambda x: (x[0].zfill(255),x[1]))
    return {'sample':S,'type':T,'header':h}

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

def get_model_hash(model_path):
    with open(model_path,'rb') as f:
        bytes = f.read()
        hash_value = hashlib.md5(bytes).hexdigest()
        return hash_value

#change to show the maximal label =>
def pred_from_vca(data,models,vca,hs={1:0.5,2:1.0},w=100,flank=0.5,max_w=100,update_pos=True,verbose=False):
    C,sv_idx = [],1
    lr_mod = keras.models.load_model(models['LR'])  #scanning model
    frames = int(lr_mod.input.shape[1]//5)
    s_i = [[0*frames,1*frames],[1*frames,2*frames],[2*frames,3*frames],[3*frames,4*frames],[4*frames,5*frames]]
    lr_n = int(lr_mod.input.shape[1])
    for vc in vca:
        if verbose: print('starting vc=%s'%vc)
        start,end,sv_len,a_id = vc[1],vc[2],vc[3],vc[-1]
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
        lr_pred = lr_mod.predict(local)
        #------------------------------------------------------------------------------------------------------------
        lr_max = [0.0,0,0]
        for c in range(1,lr_pred.shape[1],1):
            idx = np.argmax(lr_pred[:,c])
            if lr_pred[idx,c]>lr_max[0]: lr_max = [lr_pred[idx,c],idx,c]
        lr_pos = [l_rng[0]+lr_max[1]//y_len, r_rng[0]+lr_max[1]%y_len]
        if not update_pos:
            if lr_max[2] in hs:
                C += [[vc[0],vc[1],vc[2],(vc[2]-vc[1]+1),hs[lr_max[2]],lr_max[0],2,a_id]]
        else:
            if (lr_pos[1]-lr_pos[0]+1)*w>0 and lr_max[2] in hs:
                C += [[vc[0],lr_pos[0]*w,lr_pos[1]*w,(lr_pos[1]-lr_pos[0]+1)*w,hs[lr_max[2]],lr_max[0],2,a_id]]
    return C

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
#works either as list to list or dict to dict with seq as keys
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
    des = """genotype_sv: Genotype Prediction Framework v0.1.2\nCopyright (C) 2021 Timothy James Becker\n"""
    parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--norm',     type=str,  help='normalized TensorSV hdf5 file\t[None]')
    parser.add_argument('--vcf',      type=str,  help='VCF calls to genotype\t[g1kp3.hg38.vcf.gz]')
    parser.add_argument('--true_vcf', type=str,  help='optional true VCF calls to test performance\t[None]')
    parser.add_argument('--comp_vcf', type=str, help='optional comp VCF calls to test performance\t[None]')
    parser.add_argument('--run_dir',  type=str,  help='optional training run directory\t[internal models in data directory]')
    parser.add_argument('--out_dir',  type=str,  help='pickle/VCF output directory\t[None]')
    parser.add_argument('--mod_pre',  type=str,  help='model prefix\t[all]')
    parser.add_argument('--sv_types', type=str,  help='DEL,DUP,INV,INS\t[all]')
    parser.add_argument('--max_flank',type=int,  help='maximum flanking windows to use for search\t[100]')
    parser.add_argument('--factor',   type=float,help='control factor\t[0.5]')
    parser.add_argument('--base_score',action='store_true',help='use base f1 score cuttoff instead of per label f1\t[False]')
    parser.add_argument('--gpu_num',  type=int,  help='pick one of your available logical gpus\t[None]')
    args = parser.parse_args()
    if args.norm is not None:
        norm_in = args.norm
    else: raise IOError
    if args.vcf is not None:
        vcf_in = args.vcf
    else: raise IOError
    if args.true_vcf is not None:
        true_vcf_in = args.true_vcf
    else:
        true_vcf_in = None
    if args.comp_vcf is not None:
        comp_vcf_in = args.comp_vcf
    else:
        comp_vcf_in = None
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
    if args.sv_types is not None:
        svs = args.sv_types.split(',')
    else: svs = ['DEL','DUP','INV','INS']
    if args.mod_pre is not None:
        mod_pre = args.mod_pre
    else: mod_pre = 'all'
    if args.factor is not None:     factor     = args.factor
    else:                           factor     = 0.5
    if args.max_flank is not None:  max_flank  = args.max_flank
    else:                           max_flank  = 5
    if args.base_score is not None: use_base_score = True
    else:                           use_base_score = False
    if args.gpu_num is not None:    gpu_num  = args.gpu_num
    else:                           gpu_num  = 0

    file_issue = {}
    if not os.path.exists(norm_in):     file_issue['norm_in'] = norm_in
    if not os.path.exists(vcf_in):      file_issue['vcf_in'] = vcf_in
    if not os.path.exists(true_vcf_in): file_issue['true_vcf_in'] = true_vcf_in
    if not os.path.exists(comp_vcf_in): file_issue['comp_vcf_in'] = comp_vcf_in
    if len(file_issue)>0:
        print('file paths specified are broken:%s'%file_issue)
        raise IOError

    V,T = {},{}
    if vcf_in.endswith('.vcf.gz'):
        G = geno_vcf_regions_to_dict(vcf_in)
    elif vcf_in.endswith('.pickle.gz'):
        with gzip.GzipFile(vcf_in) as f:
            G = pickle.load(f)
    else:
        print('main VCF file for genotyping is missing...')
        file_issue['vcf_in'] = vcf_in
    for sv in G:
        if sv in svs:
            V[sv] = G[sv]
    if true_vcf_in is not None:
        W = geno_vcf_regions_to_dict(vcf_in)
        for sv in W:
            if sv in svs:
                T[sv] = W[sv]
    else: file_issue['true_vcf_in'] = true_vcf_in

    #checkout the .norm.hdf5 file to search on
    f = File(norm_in,'r')
    sm   = list(f.keys())[0]
    rg   = list(f[sm].keys())[0]
    seqs = sorted(list(f[sm][rg].keys()),key=lambda x: x.zfill(255))
    if type(f[sm][rg][seqs[0]]) is h5py._hl.dataset.Dataset:
        bw = f[sm][rg][seqs[0]].attrs['w']
    else:
        bw = int(sorted(f[sm][rg][seqs[0]],key=lambda x: int(x))[0]) # take smallest window size
    P = {}
    start = time.time()
    with tf.device('/gpu:%s'%gpu_num):
        for sv in V:
            if sv in svs:
                P[sv] = {}
                print('working on sv_type=%s-----------------------------------------------------------------------'%sv)
                model_path  = run_dir+'/test/models/all-%s-LR.model.hdf5'%sv
                if os.path.exists(model_path):
                    five_models = {'LR': run_dir+ '/test/models/%s-%s-LR.model.hdf5'%(mod_pre,sv)}
                else:
                    five_models = {'LR': run_dir+ '/models/%s-%s-LR.model.hdf5'%(mod_pre,sv)}
                model_base = five_models[sorted(five_models)[0]].rsplit('/')[-1].rsplit('.model.hdf5')[0]
                score_path = run_dir+'/test/scores/%s.score.json'%model_base
                if os.path.exists(score_path):
                    with open(score_path,'r') as sf: score_data = json.load(sf)
                else:
                    score_path = run_dir+'/scores/%s.score.json'%model_base
                    with open(score_path,'r') as sf: score_data = json.load(sf)
                labels  = score_data['params']['labels']
                classes = len(labels)
                base_score = float(score_data['score'])
                if 'metrics' in score_data:
                    f1_dict    = score_data['metrics'][2]
                    label_f1   = {int(x):1.0-(1.0-float(f1_dict[x]))*factor for x in f1_dict}
                else:
                    label_f1   = {labels[l]:base_score for l in labels}
                geno_bins = []
                for l in labels:
                    sv_type = l.rsplit('.')[0]
                    geno    = float('.'.join(l.rsplit('.')[1:]))
                    if sv_type!='NOT': geno_bins += [geno]
                geno_bins = sorted(geno_bins)
                hs    = {}
                for i in range(1,classes,1): hs[i] = sorted(geno_bins)[i-1]
                sh = {hs[l]:l for l in hs}
                for seq in sorted(V[sv],key=lambda x: x.zfill(255))[::-1]:
                    P[sv][seq] = []
                    if seq in f[sm][rg]:
                        seq_start = time.time()
                        print('working on seq=%s'%seq)
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
                        if   sv=='DEL': flank,max_w = 0.5,1*max_flank
                        elif sv=='DUP': flank,max_w = 0.5,2*max_flank
                        elif sv=='INV': flank,max_w = 0.5,1*max_flank
                        else:           flank,max_w = 0.0,0*max_flank
                        print('starting search on %s regions'%len(V[sv][seq]))
                        PS = pred_from_vca(data,five_models,V[sv][seq],
                                           hs=hs,w=bw,flank=flank,max_w=max_w,update_pos=False,verbose=False)

                        if use_base_score:
                            for row in PS:
                                if row[5]>=base_score: P[sv][seq] += [row]
                            print('using base score=%s, %s >= %s' %(base_score,len(PS),len(P[sv][seq])))
                        else:
                            for row in PS:
                                if row[5]>=label_f1[sh[row[4]]]: P[sv][seq] += [row]
                            print('using label f1=%s, %s >= %s' %(label_f1[sh[row[4]]],len(PS),len(P[sv][seq])))
                        seq_stop = time.time()
                        print('seq=%s was processed in %s sec'%(seq,round(seq_stop-seq_start,2)))
    f.close()
    stop = time.time()

    if true_vcf_in is not None:
        if true_vcf_in.endswith('.pickle.gz'):
            with gzip.GzipFile(true_vcf_in,'rb') as true_f:
                D,T = pickle.load(true_f),{}
                for sv in V:
                    if sv in D and sv in svs:
                        T[sv] = {}
                        for seq in V[sv]:
                            T[sv][seq] = []
                            for row in D[sv]:
                                if row[0]==seq: T[sv][seq] += [row]
                        T[sv][seq] = sorted(T[sv][seq])
                    else: T[sv] = {}
        else: print('unprocessed true VCF not currently supported...')

    if comp_vcf_in is not None:
        if comp_vcf_in.endswith('.pickle.gz'):
            with gzip.GzipFile(comp_vcf_in,'rb') as comp_f:
                E,ST = pickle.load(comp_f)['sample'],{}
                E = E[sorted(E)[0]]
                for sv in V:
                    if sv in E and sv in svs:
                        ST[sv] = {}
                        for seq in V[sv]:
                            ST[sv][seq] = []
                            for row in E[sv]:
                                if row[0]==seq: ST[sv][seq] += [row]
                        ST[sv][seq] = sorted(ST[sv][seq])
                    else: ST[sv] = {}
        else: print('unprocessed comp VCF not currently supported')

    alpha = 0.05
    S,PS = {},{}
    for sv in P:
        S[sv] = []
        for seq in P[sv]:
            S[sv] += [vc for vc in P[sv][seq]]
        #filter out class 1.0?
        S[sv] = sorted(S[sv],key=lambda x: x[5])[::-1]
        x = 0.0
        for i in range(len(S[sv])):
            x += 1.0-S[sv][i][5]
            if x>=alpha: break
        if len(S[sv])>0: S[sv] = S[sv][:i+1]
        PS[sv] = {}
        for row in S[sv]:
            seq = row[0]
            if seq in PS[sv]: PS[sv][seq] += [row]
            else:             PS[sv][seq]  = [row]
        for seq in PS[sv]:
            PS[sv][seq] = sorted(PS[sv][seq],key=lambda x: (x[0].zfill(255),x[1]))

    if true_vcf_in is not None:
        J = {sm:{}}
        for sv in T:
            J[sm][sv] = {}
            J[sm][sv]['geno_score'] = sv_score(T[sv],P[sv],r=0.9)
            J[sm][sv]['alpha=0.05'] = sv_score(T[sv],PS[sv],r=0.9)
            if comp_vcf_in is not None:
                J[sm][sv]['comp'] = sv_score(T[sv],ST[sv],r=0.9)

            print('scoring %s ---------------------------------'%sv)
            print('geno_score is:%s'%J[sm][sv]['geno_score'])
            print('alpha=0.05 is:%s'%J[sm][sv]['alpha=0.05'])
            if comp_vcf_in is not None:
                print('comp_score is:%s'%J[sm][sv]['comp'])
        print('total run completed in %s sec'%(round(stop-start,2)))
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        with open(out_dir+'%s.tensorsv.geno.json'%sm,'w') as w_f:
            json.dump(J,w_f)
            print('wrote score results to : %s/%s.tensorsv.geno.json'%(out_dir,sm))

    models = {}
    for sv in P:
        model_path  = run_dir+'/test/models/all-%s-LR.model.hdf5'%sv
        if os.path.exists(model_path):
            models[sv] = run_dir+ '/test/models/%s-%s-LR.model.hdf5'%(mod_pre,sv)
        else:
            models[sv] = run_dir+ '/models/%s-%s-LR.model.hdf5'%(mod_pre,sv)
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    write_vcf(P,models,sm,vcf_in,out_dir)
