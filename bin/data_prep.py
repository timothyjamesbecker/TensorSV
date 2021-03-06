#!/usr/env/bin/python3
import os
import gzip
import glob
import json
import copy
import time
import argparse
try:
    import cPickle as pickle
except Exception as E:
    import pickle
    pass
from h5py import File
import numpy as np
import multiprocessing as mp
import core
from hfm import hfm

def g1kp3_vcf_no_samples(vcf_path,out_path=None,c_ints=[150,150,50,50],
                         skip='#',delim='\t',info_i=7,gz=True):
    base = vcf_path.split('/')[-1].split('.vcf')[0]
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
    c_pos,c_end,c_pos95,c_end95 = c_ints
    G = []
    for i in range(len(g)):
        if not g[i][0].startswith('chr'): g[i][0] = 'chr'+g[i][0]
        sv_type = g[i][info_i].rsplit('SVTYPE=')[-1].rsplit(';')[0]
        end = g[i][info_i].rsplit(';END=')[-1].rsplit(';')[0]
        g[i][info_i] = 'SVTYPE=%s;END=%s;CIPOS=-%s,%s;CIEND=-%s,%s;CIPOS95=-%s,%s;CIEND95=-%s,%s'%\
                       (sv_type,end,c_pos,c_pos,c_end,c_end,c_pos95,c_pos95,c_end95,c_end95)
        if sv_type in ['DEL','DUP','INV']: G += [g[i]]

    h[-1] = '\t'.join(h[-1].split('\t')[:info_i+1]) #trim off the format and sample name
    s = '\n'.join(h)+'\n'
    s += '\n'.join(['\t'.join(row[0:info_i+1]) for row in G])+'\n' #cuts off the format and geno columns
    if gz:
        with gzip.GzipFile(out_path+'/%s.vcf.gz'%base,'wb') as f:
            f.write(s.encode('UTF-8'))
    else:
        with open(out_path+'/%s.vcf'%base,'w') as f:
            f.write(s)

#works on the grch38.all.geno.vcf.gz vcf file
def g1kp3_vcf_to_dict(vcf_path,out_path=None,samples='all',
                      skip='#',delim='\t',alt_i=4,info_i=7,s_i=9,geno_split='|'):
    base = vcf_path.split('/')[-1].split('.vcf')[0]
    if os.path.exists(out_path+'/%s.pickle.gz'%base):
        print('dict file was already generated, loading pickle from disk')
        with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'rb') as f:
            D = pickle.load(f)
            S,T,h = D['sample'],D['type'],D['header']
    else:
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
        if samples=='all': samples = sample_ids
        S,T = {s:{} for s in samples},{}
        row_num = 0
        for row in g:
            start = int(row[1])
            t     = row[info_i].split('SVTYPE=')[-1].split(';')[0]
            if row[alt_i].find('INS')>0:
                t,end = 'INS',start
            else:
                if row[info_i].startswith('END='):
                    end = int(row[info_i].split('END=')[1].split(';')[0])
                elif row[info_i].find('END=')>=0:
                    end = int(row[info_i].split(';END=')[1].split(';')[0])
                else:
                    end = start+1
            svl   = abs(end-start)
            if row[info_i].find('AF=')>=0:
                af = row[info_i].split(';AF=')[-1].split(';')[0]
            else: af = 1.0
            a_id  = row[2]
            if row[alt_i].find('CN')>0: #each sample only has one of the possible alleles!
                cns  = [int(x.replace('<','').replace('>','').replace('CN','')) for x in row[alt_i].split(',')]
                af   = [float(x) for x in af.split(',')]
            else:
                cns = [2 for x in row[alt_i].split(',')]
                af  = float(af)
            lt = t
            for s in sdx_i: #unpack some of the VCF file format into a dict
                gn = row[s_i:][sdx_i[s]]  #get genotype information
                ls = len(gn.split(geno_split))
                if gn=='.': gn = geno_split.join(['0' for x in range(ls)]) #check for missing values='.'
                gh = [int(x) for x in gn.split(':')[0].split(geno_split)]                #split for genotype parsing
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
        if out_path is not None:
            with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'wb') as f:
                pickle.dump({'sample':S,'type':T,'header':h},f)
    return {'sample':S,'type':T,'header':h}

#works on ALL_illumina_integrate_20170206.vcf
def hgsv_illumina_vcf_to_dict(vcf_path,out_path=None,samples='all',min_call=2,pac_bio=True,hybrid=True,
                              skip='#',delim='\t',alt_i=4,filt_i=6,info_i=7,s_i=9,geno_split='/',verbose=True):
    base = vcf_path.split('/')[-1].split('.vcf')[0]
    if os.path.exists(out_path+'/%s.pickle.gz'%base):
        print('dict file was already generated, loading pickle from disk')
        with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'rb') as f:
            D = pickle.load(f)
            S,T,H = D['sample'],D['type'],D['header']
    else:
        print('generating new dict pickle from vcf input...')
        s,g,H = '',[],[]
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
            if not t.startswith(skip):
                row = t.split(delim)
                if row[filt_i].upper()=='PASS': g += [row] #filter out lowqual
            else:                               H += [t]
        #have to scan the whole thing for sample_ids...
        C,S,T,row_num = {},{},{},0 #{samples:{types:vc}}, {type:{sample:vc}}
        for row in g:
            row_num += 1
            seq = row[0]
            vcs = [i.replace(';','') for i in row[info_i].rsplit('INFO_POS=')[-1].rsplit('INFO_END=')[0].rsplit(',')]
            pb  = row[info_i].rsplit('PB_SUP=')[-1].rsplit(';')[0]
            hy  = row[info_i].rsplit('HY_SUP=')[-1].rsplit(';')[0]
            for vc in vcs:
                vcl = vc.rsplit(':') #[start, end, type, geno, sample, caller]
                start = int(vcl[0])
                end   = int(vcl[1])
                svlen = max(1,end-start)
                if len(vcl)==6 or vcl[2]=='INS':  t = vcl[2]
                else:                             t = ':'.join(vcl[2:3+len(vcl)-6])
                sid   = vcl[-2]
                cid   = vcl[-1]
                if t =='TANDUP': t = 'DUP:TANDEM'
                if t =='DISDUP': t = 'DUP:DISPERSED'
                geno  = vcl[-3]
                if geno!='.' and geno!='.%s.'%geno_split:
                    try:
                        gt    = [int(i) for i in geno.split(geno_split)]
                        ls = len(gt)
                        if ls>=1 and gt[0]<2: #only have values 0,1 and at least one seq
                            if ls>1 or ls==1 and (seq=='chrX' or seq=='X' or seq=='chrY' or seq=='chrY'):
                                ht = sum(gt)/ls
                                if sid not in C:    C[sid] = {t:{row_num:[]}}
                                if t not in C[sid]: C[sid][t] = {row_num:[]}
                                if row_num not in C[sid][t]: C[sid][t][row_num] = []
                                if t not in T:      T[t] = {}
                                if cid not in T[t]: T[t][cid] = []
                                #[chr, start, end, sv_len, geno_type, alle_freq, copy_num, call_id,pb_str,hy_str]
                                C[sid][t][row_num] += [[seq,start,end,svlen,ht,1.0,2,cid,pb,hy]]
                                T[t][cid] += [[seq,start,end,svlen,ht,1.0,2,'%s_%s_%s'%(cid,sid,row_num)]]
                    except Exception as E: pass
            if row_num%1000==0 and verbose: print('%s rows processed'%row_num)
        if samples=='all': samples = sorted(C.keys())
        for sid in C:
            if sid in samples:
                if sid not in S: S[sid] = {}
                for t in C[sid]:
                    if t not in S[sid]: S[sid][t] = []
                    for v in C[sid][t]:
                        pb_sup,hy_sup = False,False
                        for vc in C[sid][t][v]:
                            if vc[8]!='.':
                                for pb in vc[8].rsplit(','):
                                    pb_seq,pb_start,pb_end,pb_type,pb_gt,pb_sm = pb.split(':')[:6]
                                    if sid==pb_sm and pb_type==t: pb_sup=True
                            if vc[9]!='.' and vc[9].find(':HySA')>=0:
                                hy =  vc[9].rsplit(',')[0]
                                hy_seq,hy_start,hy_end,hy_type,hy_gt = hy.split(':')[0:5]
                                hy_sms = hy.split(':')[5:-2]
                                if sid in hy_sms and hy_type==t: hy_sup=True
                        sup = ''
                        if pb_sup: sup += 'PB_'
                        if hy_sup: sup += 'HY_'
                        if len(C[sid][t][v])==1 and (min_call<=1 or (pac_bio and pb_sup) or (hybrid and hy_sup)):
                            start  = C[sid][t][v][0][1]
                            end    = C[sid][t][v][0][2]
                            svlen  = C[sid][t][v][0][3]
                            ht     = C[sid][t][v][0][4]
                            S[sid][t] += [[C[sid][t][v][0][0],start,end,svlen,ht,1.0,2,sup+'%s_%s'%('_'.join([i[7] for i in C[sid][t][v]]),v)]]
                        else: #average the results...
                            start = int(round(sum([i[1] for i in C[sid][t][v]])/len(C[sid][t][v])))
                            end   = int(round(sum([i[2] for i in C[sid][t][v]])/len(C[sid][t][v])))
                            svlen = max(1,end-start)
                            h = {}
                            if len(C[sid][t][v])>=min_call or (pac_bio and pb_sup) or (hybrid and hy_sup):
                                for i in C[sid][t][v]:
                                    if i[4] in h: h[i[4]] += 1
                                    else:         h[i[4]]  = 1
                                max_i = [0,0]
                                for i in h:
                                    if h[i]>=max_i[0]: max_i = [h[i],i]
                                ht = max_i[1] #maximal non tie-breaking
                                S[sid][t] += [[C[sid][t][v][0][0],start,end,svlen,ht,1.0,2,sup+'%s_%s'%('_'.join([i[7] for i in C[sid][t][v]]),v)]]
        for sid in S:
            for t in S[sid]:
                S[sid][t] = sorted(S[sid][t],key=lambda x: (x[0].zfill(255),x[1]))
        if out_path is not None:
            with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'wb') as f:
                pickle.dump({'sample':S,'type':T,'header':H},f)
    return {'sample':S,'type':T,'header':H}

#works on any somaCX created vcf file (including SV subtypes)
def somacx_vcf_to_dict(vcf_dir,out_path=None,min_size=25,types=['DEL','DUP','INV','INS'],sub_types=False,
                       skip='#',delim='\t',alt_i=4,info_i=7,s_i=9,geno_split='|'):
    S,T,h = {},{},[]
    sms_num = len(glob.glob(vcf_dir+'/*.vcf*'))
    row_num = 0
    for vcf_path in glob.glob(vcf_dir+'/*.vcf*'):
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
        s = '' #clear out....
        clone_ids = h[-1].split(delim)[s_i:] #genotypes are for individual clone genomes
        sm        = vcf_path.split('/')[-1].rsplit('_S0.vcf')[0].rsplit('.vcf')[0]
        print('processing sm=%s'%sm)
        S[sm] = {}
        for row in g:
            a_id  = sm+'_'+row[2]
            start = int(row[1])
            info  = row[info_i]
            t     = info.rsplit('SVTYPE=')[-1].rsplit(';')[0]
            svl   = int(info.rsplit('SVLEN=')[-1].rsplit(';')[0])
            end   = int(info.rsplit('END=')[-1].rsplit(';')[0])
            gt    = [[int(y) for y in x.rsplit(geno_split)] for x in row[s_i:]]
            ht    = sum([1.0*sum(y)/len(y) for y in gt])/len(gt)
            naf   = 1.0/sms_num
            sub_type = None
            if t=='DEL':
                cn = int(round(2.0-ht*2.0))
            elif t=='DUP': #TANDEM or DISPERSED or INV
                if info.find('DUP_TYPE=')>-1: sub_type = info.rsplit('DUP_TYPE=')[-1].rsplit(';')[0]
                else:                         sub_type = 'DISPERSED'
                if info.find('ICN=')>-1:      icn      = int(info.rsplit('ICN=')[-1].rsplit(';')[0])
                else:                         icn      = 4
                cn       = int(round(2.0 + ht*(icn-2.0)))
            elif t=='INV': #PERFECT or COMPLEX
                if info.find('INV_TYPE=')>-1: sub_type = info.rsplit('INV_TYPE=')[-1].rsplit(';')[0]
                else:                         sub_type = 'PERFECT'
                cn = 2
            else:
                cn = 2 #can come back and add cns...
            if svl>=min_size and t in types:
                if sub_types and sub_type is not None: t = '%s:%s'%(t,sub_type)
                #check to see if the INFO tag has the CALLID= tag
                if info.find('CALLID=')>-1: a_id = info.rsplit('CALLID=')[-1].rsplit(';')[0]
                if t in S[sm]: S[sm][t] += [[row[0],start,end,svl,ht,naf,cn,a_id]]
                else:          S[sm][t]  = [[row[0],start,end,svl,ht,naf,cn,a_id]]
                if t in T:
                    if a_id in T[t]: T[t][a_id][sm] = [row[0],start,end,svl,ht,naf,cn,a_id]
                    else:            T[t][a_id] = {sm:[row[0],start,end,svl,ht,naf,cn,a_id]}
                else:                T[t] = {a_id:{sm:[row[0],start,end,svl,ht,naf,cn,a_id]}}
                row_num += 1
                if row_num%100==0: print('%s rows processed'%row_num)
    if out_path is not None:
        dirs = vcf_dir.split('/')[::-1]
        for d in dirs:
            if d!='':
                base = d
                break
        with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'wb') as f:
            pickle.dump({'sample':S,'type':T,'header':h},f)
    return {'sample':S,'type':T,'header':h}

#works on any FusorSV created vcf file
def fusorsv_vcf_to_dict(vcf_dir,out_path=None,min_size=50,types=['DEL','DUP','INV'],
                        skip='#',delim='\t',alt_i=4,info_i=7,s_i=9,geno_split='/'):
    S,T,h = {},{},[]
    sms_num = len(glob.glob(vcf_dir+'/*_S-1.vcf*'))
    row_num = 0
    for vcf_path in glob.glob(vcf_dir+'/*_S-1.vcf*'):
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
        sm        = vcf_path.split('/')[-1].rsplit('_S-1.vcf')[0]
        print('processing sm=%s'%sm)
        S[sm] = {}
        for row in g:
            a_id  = sm+'_'+row[2]
            start = int(row[1])
            info  = row[info_i]
            t     = info.rsplit('SVTYPE=')[-1].rsplit(';')[0]
            svl   = int(info.rsplit('SVLEN=')[-1].rsplit(';')[0])
            end   = int(info.rsplit('END=')[-1].rsplit(';')[0])
            gt    = [[int(y) for y in x.rsplit(geno_split)] for x in row[s_i:]]
            ht    = sum([1.0*sum(y)/len(y) for y in gt])/len(gt)
            naf   = 1.0/sms_num
            cn    = 2 #can come back and add cns...
            if svl>=min_size and t in types:
                if t in S[sm]: S[sm][t] += [[row[0],start,end,svl,ht,naf,cn,a_id]]
                else:          S[sm][t]  = [[row[0],start,end,svl,ht,naf,cn,a_id]]
                if t in T:
                    if a_id in T[t]: T[t][a_id][sm] = [row[0],start,end,svl,ht,naf,cn,a_id]
                    else:            T[t][a_id] = {sm:[row[0],start,end,svl,ht,naf,cn,a_id]}
                else:                T[t] = {a_id:{sm:[row[0],start,end,svl,ht,naf,cn,a_id]}}
                row_num += 1
                if row_num%100==0: print('%s rows processed'%row_num)
    if out_path is not None:
        dirs = vcf_dir.split('/')[::-1]
        for d in dirs:
            if d!='':
                base = d
                break
        with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'wb') as f:
            pickle.dump({'sample':S,'type':T,'header':h},f)
    return {'sample':S,'type':T,'header':h}

def svtyper_sample_vcf_to_dict(vcf_path,out_path=None,
                               skip='#',delim='\t',alt_i=4,info_i=7,s_i=9,geno_split='/'):
    base = vcf_path.split('/')[-1].split('.vcf')[0]
    if os.path.exists(out_path+'/%s.pickle.gz'%base):
        print('dict file was already generated, loading pickle from disk')
        with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'rb') as f:
            D = pickle.load(f)
            S,T,h = D['sample'],D['type'],D['header']
    else:
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
        if out_path is not None:
            with gzip.GzipFile(out_path+'/%s.pickle.gz'%base,'wb') as f:
                pickle.dump({'sample':S,'type':T,'header':h},f)
    return {'sample':S,'type':T,'header':h}

def sv_regions_by_type(D,mask,seqs,flank=100,sniff_chr=True,somatic=False):
    if not somatic:
        R,M,crm = {k:[] for k in seqs},{},'%s'
        if sniff_chr: #add or take away chr if needed by sniffing a sample of rows..
            #try for the header first...
            chrs = {}
            for row in D['header']:
                if row.startswith('##contig'):
                    sq = row.rsplit('contig=<ID=')[-1].rsplit(',')[0]
                    ln = row.rsplit('length=')[-1].rsplit('>')[0]
                    chrs[sq] = ln
            if len(chrs)<=0 and len(D['sample'])>0:
                sample_data = []
                sm = list(D['sample'].keys())[0]
                i = 0
                for t in D['sample'][sm]:
                    for row in D['sample'][sm][t]:
                        sample_data += [row]
                        i += 1
                        if i > 100: break
                    if i > 100: break
                for row in sample_data:
                    if row[0] in chrs: chrs[row[0]] += 1
                    else:              chrs[row[0]]  = 1
            if len(chrs)>0:
                crm = list(chrs.keys())[0]
                if (crm not in seqs and crm not in mask) and ('chr%s'%crm in seqs and 'chr%s'%crm in mask):
                    crm = 'chr%s'
                else: crm = '%s'
            else:
                print('no samples were found in the dict...')
        for sm in D['sample']:
            M[sm] = {}
            for t in D['sample'][sm]:
                M[sm][t] = []
                for vc in D['sample'][sm][t]:
                    vc[0] = crm%vc[0]
                    if vc[0] not in mask:
                        R[vc[0]] += [[max(0,vc[1]-flank),min(vc[2]+flank,seqs[vc[0]])]]
                    else:
                        xr = [max(0,vc[1]-flank),min(vc[2]+flank,seqs[vc[0]])]
                        if core.LRF_1D(mask[vc[0]],[xr])[0]==[]:
                            R[vc[0]] += [xr]
                            M[sm][t] += [vc]
                M[sm][t] = sorted(M[sm][t],key=lambda x: (x[0].zfill(250),x[1]))
        ks,Q = [],{}
        for k in R:
            Q[k] = []
            if R[k]==[]: ks += [k]
            else:
                R[k] = sorted(R[k],key=lambda x: x[1])
                Q[k] = core.LRF_1D(R[k],R[k])[1]                  #U of all call areas (now do not have any in mask...)
                R[k] = core.LRF_1D([[0,seqs[k]]],Q[k])[2]         #D1 remove those call areas
                if k in mask: R[k] = core.LRF_1D(R[k],mask[k])[2] #D1 remove the mask areas so they are non-sv and non-mask
        for k in ks: R.pop(k)
    else:
        R,M,crm = {sm:{} for sm in D['sample']},{},'%s'
        if sniff_chr: #add or take away chr if needed by sniffing a sample of rows..
            #try for the header first...
            chrs = {}
            for row in D['header']:
                if row.startswith('##contig'):
                    sq = row.rsplit('contig=<ID=')[-1].rsplit(',')[0]
                    ln = row.rsplit('length=')[-1].rsplit('>')[0]
                    chrs[sq] = ln
            if len(chrs)<=0 and len(D['sample'])>0:
                sample_data = []
                sm = list(D['sample'].keys())[0]
                i = 0
                for t in D['sample'][sm]:
                    for row in D['sample'][sm][t]:
                        sample_data += [row]
                        i += 1
                        if i > 100: break
                    if i > 100: break
                for row in sample_data:
                    if row[0] in chrs: chrs[row[0]] += 1
                    else:              chrs[row[0]]  = 1
            if len(chrs)>0:
                crm = list(chrs.keys())[0]
                if (crm not in seqs and crm not in mask) and ('chr%s'%crm in seqs and 'chr%s'%crm in mask):
                    crm = 'chr%s'
                else: crm = '%s'
            else:
                print('no samples were found in the dict...')
        for sm in D['sample']:
            M[sm] = {}
            for t in D['sample'][sm]:
                M[sm][t] = []
                for vc in D['sample'][sm][t]:
                    vc[0] = crm%vc[0]
                    if vc[0] not in mask:
                        if vc[0] not in R[sm]: R[sm][vc[0]] = []
                        R[sm][vc[0]] += [[max(0,vc[1]-flank),min(vc[2]+flank,seqs[vc[0]])]]
                    else:
                        xr = [max(0,vc[1]-flank),min(vc[2]+flank,seqs[vc[0]])]
                        if core.LRF_1D(mask[vc[0]],[xr])[0]==[]:
                            if vc[0] not in R[sm]: R[sm][vc[0]] = []
                            R[sm][vc[0]] += [xr]
                            M[sm][t] += [vc]
                M[sm][t] = sorted(M[sm][t],key=lambda x: (x[0].zfill(250),x[1]))
        Q = {}
        for sm in R:
            Q[sm],ks = {},[]
            for k in R[sm]:
                Q[sm][k] = []
                if R[sm][k]==[]: ks += [k]
                else:
                    R[sm][k] = sorted(R[sm][k],key=lambda x: x[1])
                    Q[sm][k] = core.LRF_1D(R[sm][k],R[sm][k])[1]              #U of all call areas (could have overlaps: now do not have any in mask...)
                    R[sm][k] = core.LRF_1D([[0,seqs[k]]],Q[sm][k])[2]         #D1 remove those call areas
                    if k in mask: R[sm][k] = core.LRF_1D(R[sm][k],mask[k])[2] #D1 remove the mask areas so they are non-sv and non-mask
            for k in ks: R[sm].pop(k)
    return Q,R,M

# sm/type/vc=[seq,start,end,svlength,AF,c_id]
def write_sv_dict(M,R,out_dir,min_size=25,somatic=False,summarize=True):
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    start = time.time()
    print('generating non-sv, non-mask, AKA normal regions..')
    for sm in M:
        for t in M[sm]:
            I = []
            for i in range(len(M[sm][t])):
                if M[sm][t][i][3]>=min_size: I += [M[sm][t][i]]
            M[sm][t] = sorted(I,key=lambda x: (x[0].zfill(250),x[1]))
    if not somatic:
        # somatic must do this for each sample------------------------------------------------------
        N,j = [],1
        for k in sorted(R,key=lambda x: x.zfill(250)):
            for i in range(len(R[k])):
                if R[k][i][1]-R[k][i][0]>=min_size:
                    N += [[k,R[k][i][0],R[k][i][1],R[k][i][1]-R[k][i][0],1.0,1.0,2,'NOT_hfm_%s'%j]]
                    j += 1
        stop = time.time()
        print('finished germline in %s sec'%round(stop-start,2))
        for sm in M:
            with gzip.GzipFile(out_dir+'/%s.pickle.gz'%sm,'wb') as f:
                print('writing sample sm=%s'%sm)
                V = copy.deepcopy(M[sm])
                V['NOT'] = N
                pickle.dump(V,f)
        #somatic must do this for each sample------------------------------------------------------
    else:
        # somatic must do this for each sample------------------------------------------------------
        N,j,m = {sm:[] for sm in R},1,0
        for sm in sorted(R):
            for k in sorted(R[sm],key=lambda x: x.zfill(250)):
                for i in range(len(R[sm][k])):
                    if R[sm][k][i][1]-R[sm][k][i][0]>=min_size:
                        N[sm] += [[k,R[sm][k][i][0],R[sm][k][i][1],R[sm][k][i][1]-R[sm][k][i][0],1.0,1.0,2,'NOT_hfm_%s'%j]]
                        j += 1
                    else: m += 1
        stop = time.time()
        print('finished somatic in %s sec'%round(stop-start,2))
        for sm in M:
            if summarize:
                for t in sorted(M[sm]): print('type=%s\t%s'%(t,len(M[sm][t])))
                print('type=%s\t%s'%('NOT',len(N[sm])))
            with gzip.GzipFile(out_dir+'/%s.pickle.gz'%sm,'wb') as f:
                print('writing somatic sample sm=%s'%sm)
                V = copy.deepcopy(M[sm])
                V['NOT'] = N[sm]
                pickle.dump(V,f)
        #somatic must do this for each sample------------------------------------------------------
    return True

def get_dict_type_het(pickle_in,sort_by_len=True):
    M,N = {},{}
    with gzip.GzipFile(pickle_in,'rb') as f:
        M = pickle.load(f)
    for t in M:
        N[t] = {0.5:[],1.0:[]}
        for i in range(len(M[t])):
            if M[t][i][4]==0.5:   N[t][0.5] += [M[t][i]]
            elif M[t][i][4]==1.0: N[t][1.0] += [M[t][i]]
            else:
                if M[t][i][4] in N[t]: N[t][M[t][i][4]] += [M[t][i]]
                else:                  N[t][M[t][i][4]]  = [M[t][i]]
    if sort_by_len: #M[type][het][svlen...
        for t in N:
            for h in N[t]:
                N[t][h] = sorted(N[t][h],key=lambda x: x[3])
    else:
        for t in N: #M[type][het][coor
            for h in N[t]:
                N[t][h] = sorted(N[t][h],key=lambda x: (x[0].zfill(250),x[1]))
    return N

def dict_to_json_svtyped_regions(pickle_in,json_out,add_chr=True):
    M,R = {},{}
    with gzip.GzipFile(pickle_in,'rb') as pickle_f:
        M = pickle.load(pickle_f)
        for sm in M['sample']:
            for t in M['sample'][sm]:
                if t not in R: R[t] = {}
                for vc in M['sample'][sm][t]:
                    seq,start,end = vc[0:3]
                    if add_chr: seq = 'chr'+seq
                    reg = (seq,start,end)
                    if reg in R[t]: R[t][reg] += 1
                    else:           R[t][reg]  = 1
        J = {}
        for t in R:
            J[t] = []
            for reg in R[t]: J[t] += [[reg[0],reg[1],reg[2]]]
            J[t] = sorted(J[t],key=lambda x: (x[0].zfill(255),x[1]))
        with open(json_out,'w') as json_f:
            json.dump(J,json_f)
    return True

#given an hdf5 file, generate a dict of zero coordinates using the smallest windows...
def get_maxima_regions(hdf5_paths,out_dir,lower_cut=0.1,upper_cut=1000.0,verbose=True):
    R,out = {},'fail'
    if type(hdf5_paths) is not list: hdf5_paths = [hdf5_paths]
    for hdf5_path in hdf5_paths:
        if verbose: print('generating maxima ranges for %s'%hdf5_path.rsplit('/')[-1])
        A = hfm.get_hdf5_attrs(hdf5_path)
        sm = list(A.keys())[0]
        if not os.path.exists(out_dir+'/%s.sm.maxima.json'%sm):
            h = hfm.HFM()
            for sm in A:
                R[sm] = {}
                for rg in A[sm]:
                    R[sm][rg] = {}
                    for seq in A[sm][rg]:
                        start = time.time()
                        R[sm][rg][seq] = []
                        w = int(sorted(list(A[sm][rg][seq].keys()),key=lambda x: int(x))[0])
                        l = int(A[sm][rg][seq][str(w)]['len'])
                        m = A[sm][rg][seq][str(w)]['M1']
                        h.buffer_chunk(hdf5_path,seq=seq,start=0,end=l,
                                       sms=[sm],rgs=[rg],tracks=['total'],
                                       features=['moments'],windows=[str(w)])
                        data = h.__buffer__
                        mmts = data[sm][rg][seq]['total']['moments'][str(w)]
                        i = 0
                        while i < len(mmts):
                            j = i
                            while j < len(mmts) and (mmts[j][m]<lower_cut or mmts[j][m]>upper_cut): j += 1
                            if i<j: R[sm][rg][seq] += [[w*i,w*j]]
                            i = j+1
                        if len(R[sm][rg][seq])>0 and R[sm][rg][seq][-1][1]>l: R[sm][rg][seq][-1][1]=l
                        h.__buffer__ = None
                        stop = time.time()
                        if verbose: print('processed sm=%s rg=%s seq%s in %s secs'%(sm,rg,seq,round(stop-start,2)))
            if verbose: print('finished processing %s'%hdf5_path)
            with open(out_dir+'/%s.sm.maxima.json'%sm,'w') as f:
                json.dump(R,f)
                out = 'success'
        else:
            print('found maxima file for sm=%s, skipping calculation'%sm)
            with open(out_dir+'/%s.sm.maxima.json'%sm,'r') as f:
                R = json.load(f)
                out = 'success'
    return {'sm':sm,'success':out}

def intersect_maxima_regions_ooc(out_dir,min_w=500,verbose=True):
    I = {}
    if out_dir is not None and os.path.exists(out_dir+'/sms.maxima.json'):
        with open(out_dir+'/sms.maxima.json','r') as f:
            I = json.load(f)
    else:
        I = {}
        maxima_files = glob.glob(out_dir+'/*sm.maxima.json')
        for file in maxima_files:
            R = {}
            with open(file,'r') as f: R = json.load(f)
            sm = list(R.keys())[0]
            rg = list(R[sm].keys())[0]
            if verbose: print('intersecting, extending and filtering ranges for sample:%s'%sorted(list(R.keys())[0]))
            for seq in R[sm][rg]: I[seq] = R[sm][rg][seq] #start the first group
            for sm in R:                                  #now the rest of the sample
                for rg in R[sm]:
                    for seq in R[sm][rg]:
                        I[seq] = core.LRF_1D(I[seq],R[sm][rg][seq])[0]
        R = {}
        for seq in I:
            T = []
            for i in range(len(I[seq])): #[1] filter ranges that are too small
                if I[seq][i][1]-I[seq][i][0] >= min_w: T += [I[seq][i]]
            I[seq] = T
            for i in range(len(I[seq])-1):
                if I[seq][i+1][0]-I[seq][i][1] <= min_w: I[seq][i][1] += min_w
            I[seq] = core.LRF_1D(I[seq],I[seq])[1] #union of the gap extended sections
            I[seq] = [[int(x[0]),int(x[1])] for x in I[seq]]
            if verbose: print('%s intersected, extended, ranges passed filters for seq=%s'%(len(I[seq]),seq))
        if out_dir is not None:
            if not os.path.exists(out_dir): os.mkdir(out_dir)
            with open(out_dir+'/sms.maxima.json','w') as f:
                json.dump(I,f)
    return I

def preprocess_hfm_five_point(hfm_in,hdf5_out,
                              feature_order = ['total','primary','alternate','proper_pair','discordant','RD','GC','MD',
                                               'mapq_pp','mapq_dis','big_del','deletion','insertion','substitution','splice','fwd_rev_diff',
                                               'tlen_pp', 'tlen_pp_rd', 'tlen_dis', 'tlen_dis_rd','right_clipped','left_clipped',
                                               'orient_same','orient_out','orient_um','orient_chr',
                                               'left_smap_same','left_smap_diff','right_smap_same','right_smap_diff'],
                              final_range=[0.0,1.0],standardize=True,final_dtype='f2',compression='lzf',verbose=False):
    if not os.path.exists(hdf5_out):
        h = hfm.HFM()
        f = File(hdf5_out,'w')
        A = hfm.get_hdf5_attrs(hfm_in)
        M = hfm.get_hdf5_map(hfm_in)
        feature_issues = set([])
        for sm in A:
            if verbose: print('preprocessing sm=%s'%sm)
            for rg in A[sm]:
                if verbose: print('preprocessing sm=%s rg=%s'%(sm,rg))
                for seq in A[sm][rg]:
                    if verbose: print('preprocessing sm=%s rg=%s seq=%s'%(sm,rg,seq))
                    start = time.time()
                    data,attrs,nk = {},[],[]
                    w = int(sorted(A[sm][rg][seq],key = lambda x: int(x))[0])
                    l = A[sm][rg][seq][str(w)]['len']
                    for k in feature_order:
                        if k in M[sm][rg][seq]:
                            if verbose: print('preprocessing=>buffering sm=%s rg=%s seq=%s ftr=%s' % (sm, rg, seq, k))
                            h.buffer_chunk(hfm_in,seq,0,l,sms=[sm],rgs=[rg],tracks=[k],features=['moments'],windows=[str(w)])
                            if verbose: print('preprocessing=>std/norm sm=%s rg=%s seq=%s ftr=%s' % (sm, rg, seq, k))
                            raw = core.standardized_moments(h.__buffer__[sm][rg][seq][k]['moments'][str(w)])
                            if standardize:
                                for d in range(raw.shape[1]):
                                    d_std  = np.std(raw[:,d])
                                    d_mean = np.mean(raw[:,d])
                                    raw[:,d] -= d_mean
                                    if d_std>0.0: raw[:,d] /= d_std
                                    attrs += [[k,int(d),'mean',d_mean]]
                                    attrs += [[k,int(d),'std',d_std]]
                            if final_range is not None and len(final_range)>1:
                                offset,scale = final_range[0],final_range[1]-final_range[0]
                                for d in range(raw.shape[1]):
                                    min_v   = np.min(raw[:,d])
                                    max_v   = np.max(raw[:,d])
                                    if max_v-min_v > 0.0:
                                        raw[:,d] -= min_v
                                        raw[:,d] /= (max_v-min_v)
                                        raw[:,d] *= scale
                                        raw[:,d] += offset
                                    elif min_v > 0.0:
                                        raw[:,d] -= min_v
                                        raw[:,d] *= scale
                                        raw[:,d] += offset
                                    attrs += [[k,int(d),'min',min_v]]
                                    attrs += [[k,int(d),'max',max_v]]
                            data[k] = raw
                            if verbose: print('completed=>std/norm sm=%s rg=%s seq=%s ftr=%s' % (sm, rg, seq, k))
                        else:
                            for i in range(len(feature_order)):
                                if feature_order[i]==k:
                                    feature_issues.add(i)
                                    break
                            nk += [k]
                    ks = sorted(list(set(feature_order).difference(set(nk))))
                    if verbose: print('writing data for sm=%s rg=%s seq=%s'%(sm,rg,seq))
                    if compression=='gzip':
                        out = f.create_dataset('%s/%s/%s'%(sm,rg,seq),(data[ks[0]].shape[0],len(ks),data[ks[0]].shape[1]),
                                               dtype=final_dtype,compression='gzip',compression_opts=9,shuffle=True)
                    else:
                        out = f.create_dataset('%s/%s/%s'%(sm,rg,seq),(data[ks[0]].shape[0],len(ks),data[ks[0]].shape[1]),
                                               dtype=final_dtype,compression='lzf',shuffle=True)
                    if verbose: print('writing attrs for sm=%s rg=%s seq=%s'%(sm,rg,seq))
                    out.attrs['l']             = data[ks[0]].shape[0]
                    out.attrs['f']             = len(ks)
                    out.attrs['m']             = data[ks[0]].shape[1]
                    out.attrs['w']             = w
                    out.attrs['standardize']   = standardize
                    out.attrs['range']         = (final_range if final_range is not None else False)
                    out.attrs['dtype']         = final_dtype
                    if len(feature_issues)>0:
                        print('feature issue detected...%s'%feature_issues)
                        ftrs = []
                        for i in range(len(feature_order)):
                            if i not in feature_issues: ftrs += [feature_order[i]]
                        ftr_order = ftrs
                    else: ftr_order = feature_order
                    out.attrs['feature_order'] = ftr_order
                    #could keep d1,d2,d3,d4 range and std norm to reverse
                    out.attrs['trans_hist']    = attrs
                    # with open(hdf5_out.replace('.hdf5','.seq-%s.trans.hist'%seq),'w') as file:
                    #     file.write('\n'.join(['\t'.join([str(x) for x in row]) for row in attrs]))
                    for i in range(len(ftr_order)): out[:,i,:] = data[ftr_order[i]][:,:]
                    stop = time.time()
                    print('sm=%s rg=%s seq=%s processed in %s sec'%(sm,rg,seq,round(stop-start,2)))
        f.close()
    else:
        print('found hdf5_out file=%s'%hdf5_out)
    return True

#hdf5_in=hfm file path for input, target_in=g1kp3_dict.pickle.gz, track_order=['total','GC','discordant',...]
#hdf5_out output file path,w = window size to use, n_w how many windows to flank on the breakpoints
#if not_prop is proportion of NOT, where 1.0 is same as the genome (IE very high: NOT:55E3 to DEL:1.5E3, etc...
def five_point_targets(hdf5_in,target_in,hdf5_out,types=['DEL','DUP','CNV','INV','INS','NOT'],
                        w_shift={'DEL':[-2,-1,0,1,2],'INS':[-2,-1,0,1,2],
                                 'DUP':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'DUP:DISPERSED':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
                                 'DUP:TANDEM':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'DUP:INV':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
                                 'INV':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'INV:PERFECT':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
                                 'INV:COMPLEX':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'CNV':[-2,-1,0,1,2],'NOT':[0]},
                        n_w=5,not_prop=0.5,compression='gzip',verbose=False):
    T = get_dict_type_het(target_in,sort_by_len=True) #T[type][het][0] = [seq,start,end,svlen,het,af,cn,a_id]
    S = {}
    if not os.path.exists(hdf5_out):
        f = File(hdf5_in,'r')
        g = File(hdf5_out,'w')
        for sm in f:
            for rg in f[sm]:
                for seq in f[sm][rg]:
                    attrs = f[sm][rg][seq].attrs
                    if len(attrs)<1:
                        w = sorted(f[sm][rg][seq],key=lambda x: int(x))[0]
                        attrs = f[sm][rg][seq][w].attrs
                    S[seq] = {a:attrs[a] for a in attrs} #should be uniform for w,m,f
        for sm in f:
            for rg in f[sm]:
                for t in types:
                    if t in T: #special routines for NOT areas?
                        for h in T[t]:
                            t_h_start = time.time()
                            j,x,V = 0,2*n_w+1,[] #j is number of instances processed
                            if t!='NOT':
                                for i in range(len(T[t][h])):
                                    seq = T[t][h][i][0]
                                    if seq in f[sm][rg]:
                                        w = S[seq]['w']
                                        if T[t][h][i][0] in f[sm][rg]: #if seq in f[sm][rg]
                                            for s in w_shift[t]:
                                                a,b = T[t][h][i][1]+int(1.5*w*s),T[t][h][i][2]+int(1.5*w*s)
                                                V += [[T[t][h][i][0],a,b,b-a]+T[t][h][i][4:]]
                                                j += 1
                                print('sm=%s using %s sv targets of type %s:%s'%(sm,j,t,h))
                            else: #'background region has around 55E3 areas...with 29E3 that will pass the filter below
                                c1 = 0 #progress counter
                                if len(T[t][h])>0:
                                    for i in range(len(T[t][h])):
                                        seq = T[t][h][i][0]
                                        if seq in S:
                                            seq_l,w = S[seq]['l'],S[seq]['w']
                                            a,b = min(T[t][h][i][1]-2*x,seq_l*w),min(T[t][h][i][2]-x,seq_l*w)
                                            if seq in f[sm][rg] and b-a>2*(2*n_w+1)*w:
                                                a  = np.random.choice(range(a+w*n_w,b-2*w*n_w,1),1)[0]
                                                b  = np.random.choice(range(b-2*w*n_w,b-w*n_w,1),1)[0]
                                                V += [[seq,a,b,b-a]+T[t][h][i][4:]]
                                                j += 1
                                            c1 += 1
                                            if int(len(T[t][h])>10) and c1%(int(len(T[t][h])/10.0))==0:
                                                print('sm=%s generated %s random background targets (NOT) out of %s'%(sm,j,c1))
                                    r,U = 0,[]
                                    v_idx = np.random.choice(range(len(V)),min(len(V),int(round(not_prop*len(V)))),replace=False)
                                    if len(v_idx)<len(V):
                                        for i in v_idx:
                                            U += [V[i]]
                                            r += 1
                                        V = U
                                        j = r
                                    print('sm=%s randomly sampled %s background targets (NOT)'%(sm,j))
                            #check dimensions for each sequence and remove those from V that don't match S?
                            if len(V)>0:
                                print('sm=%s adding t=%s:h=%s to the dataset'%(sm,t,h))
                                seq_m,bad_seq = max([S[seq]['f'] for seq in S]),[]
                                for seq in S:
                                    if S[seq]['f']!=seq_m: bad_seq += [seq]
                                if len(bad_seq)>0:
                                    print('bad_seq=%s, removing incompatible labels'%bad_seq)
                                    W = []
                                    for i in range(len(V)):
                                        if V[i][0] not in bad_seq: W += [V[i]]
                                    V = W
                                if compression=='gzip':
                                    out = g.create_dataset('%s/%s.%s'%(sm,t,h),(j,5*x,S[list(S.keys())[0]]['f'],S[list(S.keys())[0]]['m']),
                                                           dtype=S[list(S.keys())[0]]['dtype'],shuffle=True,compression='gzip',
                                                           compression_opts=9)
                                else:
                                    out = g.create_dataset('%s/%s.%s'%(sm,t,h),(j,5*x,S[list(S.keys())[0]]['f'],S[list(S.keys())[0]]['m']),
                                                           dtype=S[list(S.keys())[0]]['dtype'],shuffle=True,compression='lzf')
                                attrs = {'a_id':[],'cn':[],'svlen':[]}
                                data = np.zeros((5*x,S[list(S.keys())[0]]['f'],S[list(S.keys())[0]]['m']),dtype=S[list(S.keys())[0]]['dtype'])
                                print('sm=%s writing targets and attributes to the dataset for t=%s:h=%s'%(sm,t,h))
                                c1,w = 0,S[list(S.keys())[0]]['w']
                                for i in range(len(V)):
                                    seq,seq_l,start,end = V[i][0],S[V[i][0]]['l'],V[i][1],V[i][2]
                                    l,r                 = int(round(start/w)),int(round(end/w))
                                    a,m,b               = max(n_w,int(round(l-(r-l)/2.0))),int(round(l+(r-l)/2.0)),\
                                                          min(int(round(r+(r-l)/2.0)),seq_l-n_w-1)
                                    regs                = [a,l,m,r,b]
                                    attrs['svlen']     += [V[i][3]+1]
                                    attrs['cn']        += [V[i][6]]
                                    attrs['a_id']      += [V[i][7]]
                                    if len(f[sm][rg][seq].attrs)>0:
                                        for y in range(len(regs)):
                                            data[y*x:(y+1)*x,:,:] = f[sm][rg][seq][regs[y]-n_w:regs[y]+n_w+1,:,:]
                                    else:
                                        for y in range(len(regs)):
                                            data[y*x:(y+1)*x,:,:] = f[sm][rg][seq][str(w)][regs[y]-n_w:regs[y]+n_w+1,:,:]
                                    out[i,:,:,:] = data[:,:,:]
                                    c1 += 1
                                    if int(len(V)>10) and c1%(int(len(V)/10.0))==0:
                                        print('sm=%s finished %s targets (%s:%s) of %s'%(sm,c1,t,h,len(V)))
                                for k in S[seq]: out.attrs[k] = S[seq][k]
                                # if t!='NOT': #conversion/approximation of svlen and cn to uint8 -> [0,255]
                                try:
                                    attrs['svlen'] = np.clip(np.round(12.75*np.log(attrs['svlen'])),a_min=0,a_max=np.iinfo(np.uint8).max)
                                    out.attrs.create('svlen',attrs['svlen'],dtype=np.uint8)
                                except Exception as E:
                                    print('total size of 8-bit svlen attributes %s beyond the h5py limits...'%len(attrs['svlen']))
                                    pass
                                try:
                                    attrs['cn'] = np.clip(attrs['cn'],a_min=0,a_max=np.iinfo(np.uint8).max)
                                    out.attrs.create('cn',attrs['cn'],dtype=np.uint8)
                                except Exception as E:
                                    print('total size of 8-bit cn attributes %s beyond the h5py limits...'%len(attrs['svlen']))
                                    pass
                                    # out.attrs['a_id']  = attrs['a_id'] #variable strings are not the best for hdf5 files...
                                t_h_stop = time.time()
                                print('sm=%s finished t=%s:h=%s in %s sec\n'%(sm,t,h,round(t_h_stop-t_h_start,2)))
                            else:
                                print('sm=%s skipping t=%s:h=%s, no targets present'%(sm,t,h))
        f.close()
        g.close()
    else:
        print('%s was already processed, skipping'%hdf5_out)
    return True

def preprocess_hfm_zoom(hfm_in,hdf5_out,
                        feature_order = ['total','primary','alternate','proper_pair','discordant','RD','GC','MD',
                                         'mapq_pp','mapq_dis','big_del','deletion','insertion','substitution','splice','fwd_rev_diff',
                                         'tlen_pp', 'tlen_pp_rd', 'tlen_dis', 'tlen_dis_rd','right_clipped','left_clipped',
                                         'orient_same','orient_out','orient_um','orient_chr',
                                         'left_smap_same','left_smap_diff','right_smap_same','right_smap_diff'],
                        final_range=None,standardize=True,final_dtype='f2',compression='lzf',verbose=False):
    if not os.path.exists(hdf5_out):
        h = hfm.HFM()
        f = File(hdf5_out,'w')
        A = hfm.get_hdf5_attrs(hfm_in)
        M = hfm.get_hdf5_map(hfm_in)
        feature_issues = set([])
        for sm in A:
            if verbose: print('preprocessing sm=%s'%sm)
            for rg in A[sm]:
                if verbose: print('preprocessing sm=%s rg=%s'%(sm,rg))
                for seq in A[sm][rg]:
                    if verbose: print('preprocessing sm=%s rg=%s seq=%s'%(sm,rg,seq))
                    start = time.time()
                    ws = [int(w) for w in sorted(A[sm][rg][seq],key = lambda x: int(x))]
                    for w in ws:
                        data,attrs,nk = {},[],[]
                        l = A[sm][rg][seq][str(w)]['len']
                        for k in feature_order:
                            if k in M[sm][rg][seq]:
                                h.buffer_chunk(hfm_in,seq,0,l,sms=[sm],rgs=[rg],tracks=[k],features=['moments'],windows=[str(w)])
                                raw = core.standardized_moments(h.__buffer__[sm][rg][seq][k]['moments'][str(w)])
                                if standardize:
                                    for d in range(raw.shape[1]):
                                        d_std  = np.std(raw[:,d])
                                        d_mean = np.mean(raw[:,d])
                                        raw[:,d] -= d_mean
                                        if d_std>0.0: raw[:,d] /= d_std
                                        attrs += [[k,int(d),'mean',d_mean]]
                                        attrs += [[k,int(d),'std',d_std]]
                                if final_range is not None and len(final_range)>1:
                                    offset,scale = final_range[0],final_range[1]-final_range[0]
                                    for d in range(raw.shape[1]):
                                        min_v   = np.min(raw[:,d])
                                        max_v   = np.max(raw[:,d])
                                        if max_v-min_v > 0.0:
                                            raw[:,d] -= min_v
                                            raw[:,d] /= (max_v-min_v)
                                            raw[:,d] *= scale
                                            raw[:,d] += offset
                                        elif min_v > 0.0:
                                            raw[:,d] -= min_v
                                            raw[:,d] *= scale
                                            raw[:,d] += offset
                                        attrs += [[k,int(d),'min',min_v]]
                                        attrs += [[k,int(d),'max',max_v]]
                                data[k] = raw
                            else:
                                for i in range(len(feature_order)):
                                    if feature_order[i]==k:
                                        feature_issues.add(i)
                                        break
                                nk += [k]
                        ks = sorted(list(set(feature_order).difference(set(nk))))
                        if verbose: print('writing data for sm=%s rg=%s seq=%s'%(sm,rg,seq))
                        if compression=='gzip':
                            out = f.create_dataset('%s/%s/%s/%s'%(sm,rg,seq,w),(data[ks[0]].shape[0],len(ks),data[ks[0]].shape[1]),
                                                   dtype=final_dtype,compression='gzip',compression_opts=9,shuffle=True)
                        else:
                            out = f.create_dataset('%s/%s/%s/%s'%(sm,rg,seq,w),(data[ks[0]].shape[0],len(ks),data[ks[0]].shape[1]),
                                                   dtype=final_dtype,compression='lzf',shuffle=True)
                        out.attrs['l']             = data[ks[0]].shape[0]
                        out.attrs['f']             = len(ks)
                        out.attrs['m']             = data[ks[0]].shape[1]
                        out.attrs['w']             = w
                        out.attrs['standardize']   = standardize
                        out.attrs['range']         = (final_range if final_range is not None else False)
                        out.attrs['dtype']         = final_dtype
                        if len(feature_issues)>0:
                            print('feature issue detected...%s'%feature_issues)
                            ftrs = []
                            for i in range(len(feature_order)):
                                if i not in feature_issues: ftrs += [feature_order[i]]
                            ftr_order = ftrs
                        else: ftr_order = feature_order
                        out.attrs['feature_order'] = ftr_order
                        #could keep d1,d2,d3,d4 range and std norm to reverse?
                        out.attrs['trans_hist']    = attrs
                        # with open(hdf5_out.replace('.hdf5','.seq-%s.trans.hist'%seq),'w') as file:
                        #     file.write('\n'.join(['\t'.join([str(x) for x in row]) for row in attrs]))
                        for i in range(len(ftr_order)): out[:,i,:] = data[ftr_order[i]][:,:]
                        stop = time.time()
                        print('sm=%s rg=%s seq=%s w=%s processed in %s sec'%(sm,rg,seq,w,round(stop-start,2)))
                        # for k in data: print(np.mean(data[k]))
        f.close()
    else:
        print('found hdf5_out file=%s'%hdf5_out)
    return True

def get_closest_w(vc,S,max_size=44,target_size=33):
    if vc[0] in S:
        ws = sorted([int(s) for s in S[vc[0]]])
        best = [0,max_size,max_size]
        for i in range(len(ws)):
            n = vc[3]//ws[i]
            if n<max_size and abs(n-target_size)<best[2]: best = [i,n,abs(n-target_size)]
        return [ws[best[0]],best[1]]
    else:
        return []

def zoom_targets(hdf5_in,target_in,hdf5_out,types=['DEL','DUP','CNV','INV','INS','NOT'],
                        w_shift={'DEL':[-2,-1,0,1,2],'INS':[-2,-1,0,1,2],
                                 'DUP':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'DUP:DISPERSED':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
                                 'DUP:TANDEM':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'DUP:INV':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
                                 'INV':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'INV:PERFECT':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
                                 'INV:COMPLEX':[-5,-4,-3,-2,-1,0,1,2,3,4,5],'CNV':[-2,-1,0,1,2],'NOT':[0]},
                        size=55,lim_size=44,target_size=33,not_prop=1.0,compression='lzf',verbose=False):
    T = get_dict_type_het(target_in,sort_by_len=True) #T[type][het][0] = [seq,start,end,svlen,het,af,cn,a_id]
    S = {}
    if not os.path.exists(hdf5_out):
        f = File(hdf5_in,'r')
        g = File(hdf5_out,'w')
        for sm in f:
            for rg in f[sm]:
                for seq in f[sm][rg]:
                    S[seq] = {}
                    for w in f[sm][rg][seq]:
                        attrs = f[sm][rg][seq][w].attrs
                        S[seq][w] = {a:attrs[a] for a in attrs} #should be uniform for w,m,f
        for sm in f:
            for rg in f[sm]:
                for t in types:
                    if t in T: #special routines for NOT areas?
                        for h in T[t]:
                            t_h_start = time.time()
                            j,V = 0,[] #j is number of instances processed
                            if t!='NOT':
                                for i in range(len(T[t][h])):
                                    seq = T[t][h][i][0]
                                    if seq in f[sm][rg]:
                                        start,end,svlen = T[t][h][i][1],T[t][h][i][2],T[t][h][i][3]
                                        best = get_closest_w(T[t][h][i],S,lim_size,target_size)
                                        w = best[0]
                                        for s in w_shift[t]:
                                            a,b = start+w*s,end+w*s
                                            V += [[seq,a,b,b-a]+T[t][h][i][4:-1]+[T[t][h][i][-1]+'_shift_%sbp'%(w*s)]]
                                            j += 1
                                print('sm=%s using %s sv targets of type %s:%s'%(sm,j,t,h))
                            else: #'background region has around 55E3 areas...with 29E3 that will pass the filter below
                                c1 = 0 #progress counter
                                if len(T[t][h])>0:
                                    for i in range(len(T[t][h])):
                                        seq = T[t][h][i][0]
                                        if seq in f[sm][rg]:
                                            start,end,svlen = T[t][h][i][1],T[t][h][i][2],T[t][h][i][3]
                                            best = get_closest_w(T[t][h][i],S,lim_size,target_size)
                                            w = best[0]
                                            for s in w_shift[t]:
                                                a,b = start+w*s,end+w*s
                                                V += [[seq,a,b,b-a]+T[t][h][i][4:-1]+[T[t][h][i][-1]+'_shift_%sbp'%(w*s)]]
                                                j += 1
                                            c1 += 1
                                            if int(len(T[t][h])>10) and c1%(int(len(T[t][h])/10.0))==0:
                                                print('sm=%s generated %s random background targets (NOT) out of %s'%(sm,j,c1))
                                    r,U = 0,[]
                                    v_idx = np.random.choice(range(len(V)),min(len(V),int(round(not_prop*len(V)))),replace=False)
                                    if len(v_idx)<len(V):
                                        for i in v_idx:
                                            U += [V[i]]
                                            r += 1
                                        V = U
                                        j = r
                                    print('sm=%s randomly sampled %s background targets (NOT)'%(sm,j))
                            if len(V)>0:
                                print('sm=%s adding t=%s:h=%s to the dataset'%(sm,t,h))
                                seq = sorted(S)[0]
                                if compression=='gzip':
                                    out = g.create_dataset('%s/%s.%s'%(sm,t,h),(j,size,S[seq][sorted(S[seq])[0]]['f'],S[seq][sorted(S[seq])[0]]['m']),
                                                           dtype=S[seq][sorted(S[seq])[0]]['dtype'],shuffle=True,compression='gzip',
                                                           compression_opts=9)
                                else:
                                    out = g.create_dataset('%s/%s.%s'%(sm,t,h),(j,size,S[seq][sorted(S[seq])[0]]['f'],S[seq][sorted(S[seq])[0]]['m']),
                                                           dtype=S[seq][sorted(S[seq])[0]]['dtype'],shuffle=True,compression='lzf')

                                #h5py attr limits make it difficult to store a_ids...IDEA: json file version of this?
                                #JSON a_id file: {sm:{t:{h:[a_id_1,a_id_2,... a_id_n] : a_id_x is a string data type
                                attrs = {'a_id':[],'cn':[],'svlen':[]}
                                data = np.zeros((size,S[seq][sorted(S[seq])[0]]['f'],S[seq][sorted(S[seq])[0]]['m']),
                                                dtype=S[seq][sorted(S[seq])[0]]['dtype'])

                                print('sm=%s writing targets and attributes to the dataset for t=%s:h=%s'%(sm,t,h))
                                c1 = 0
                                for i in range(len(V)):
                                    seq,start,end,svlen = V[i][0],V[i][1],V[i][2],V[i][3]
                                    best = get_closest_w(V[i],S,lim_size,target_size)
                                    w = best[0]
                                    a = max(0,int(round(int(round(start/w))+(int(round(end/w))-int(round(start/w)))/2.0))-size//2)
                                    if a+size>=f[sm][rg][seq][str(w)].shape[0]:
                                        print('hit boundry skipping:V=%s,w=%s,a=%s'%(V[i],w,a))
                                    else:
                                        a = (f[sm][rg][seq][str(w)].shape[0]-size-1 if a+size>=f[sm][rg][seq][str(w)].shape[0] else a)
                                        attrs['svlen']     += [V[i][3]+1]
                                        attrs['cn']        += [V[i][6]]
                                        # attrs['a_id']      += [V[i][7]]
                                        data[:,:,:]  = f[sm][rg][seq][str(w)][a:a+size,:,:]
                                        out[i,:,:,:] = data[:,:,:]
                                        c1 += 1
                                        if int(len(V)>10) and c1%(int(len(V)/10.0))==0:
                                            print('sm=%s finished %s targets (%s:%s) of %s'%(sm,c1,t,h,len(V)))
                                for k in S[seq][str(w)]: out.attrs[k] = S[seq][str(w)][k]
                                # if t!='NOT': #conversion/approximation of svlen and cn to uint8 -> [0,255]
                                try:
                                    attrs['svlen'] = np.clip(np.round(12.75*np.log(attrs['svlen'])),a_min=0,a_max=np.iinfo(np.uint8).max)
                                    out.attrs.create('svlen',attrs['svlen'],dtype=np.uint8)
                                except Exception as E:
                                    print('total size of 8-bit svlen attributes %s beyond the h5py limits...'%len(attrs['svlen']))
                                    pass
                                try:
                                    attrs['cn'] = np.clip(attrs['cn'],a_min=0,a_max=np.iinfo(np.uint8).max)
                                    out.attrs.create('cn',attrs['cn'],dtype=np.uint8)
                                except Exception as E:
                                    print('total size of 8-bit cn attributes %s beyond the h5py limits...'%len(attrs['svlen']))
                                    pass
                                    # out.attrs['a_id']  = attrs['a_id'] #variable strings are not the best for hdf5 files...
                                t_h_stop = time.time()
                                print('sm=%s finished t=%s:h=%s in %s sec\n'%(sm,t,h,round(t_h_stop-t_h_start,2)))
                            else:
                                print('sm=%s skipping t=%s:h=%s, no targets present'%(sm,t,h))
        f.close()
        g.close()
    else:
        print('%s was already processed, skipping'%hdf5_out)
    return True
def w_shift(shift=2):
    return [i for i in range(-1*shift,0,1)]+[0]+[i for i in range(1,shift+1,1)]

def process_labels(sm,hfm_in,hdf5_out,target_in,hdf5_label,types,p,n_w,stand,rng,comp,zoom,shift_w,verbose):
    out,start = '',time.time()
    wind_shift = {'DEL':w_shift(shift_w),'INS':w_shift(shift_w),
                  'DUP':w_shift(2*shift_w+1),'DUP:DISPERSED':w_shift(2*shift_w+1),
                  'DUP:TANDEM':w_shift(2*shift_w+1),'DUP:INV':w_shift(2*shift_w+1),
                  'INV':w_shift(2*shift_w+1),'INV:PERFECT':w_shift(2*shift_w+1),
                  'INV:COMPLEX':w_shift(2*shift_w+1),'CNV':w_shift(shift_w),'NOT':w_shift(shift_w-2)}
    if args.somacx_vcf:
        wind_shift['NOT'] = w_shift(2*shift_w+1)
    elif args.hgsv_vcf:
        wind_shift = {'DEL':w_shift(shift_w-1),'INS':w_shift(shift_w-1),'DUP':w_shift(shift_w),
                      'INV':w_shift(shift_w),'NOT':w_shift(shift_w-2)}
    if not zoom:
        try:
            if stand: print('starting normalization on sm=%s with standardization'%sm)
            else:     print('starting normalization on sm=%s'%sm)
            preprocess_hfm_five_point(hfm_in,hdf5_out,final_range=rng,standardize=stand,final_dtype='f2',compression=comp,verbose=verbose)
            print('starting five point target acquisition on sm=%s, hdf5_in=%s, target_in=%s label_out=%s'%(sm,hdf5_out,target_in,hdf5_label))
            five_point_targets(hdf5_out,target_in,hdf5_label,types=types,n_w=n_w,not_prop=p,w_shift=wind_shift,compression=comp,verbose=verbose)
        except Exception as E: out = E
    else:
        try:
            if stand: print('starting zoom normalization on sm=%s with standardization'%sm)
            else:     print('starting zoom normalization on sm=%s'%sm)
            preprocess_hfm_zoom(hfm_in,hdf5_out,final_range=rng,standardize=stand,final_dtype='f2',compression=comp,verbose=verbose)
            print('starting zoom target acquisition on sm=%s, hdf5_in=%s, target_in=%s label_out=%s'%(sm,hdf5_out,target_in,hdf5_label))
            zoom_targets(hdf5_out,target_in,hdf5_label,types=types,w_shift=wind_shift,
                         size=n_w*11,lim_size=n_w*9,target_size=n_w*7,not_prop=p,compression=comp,verbose=verbose)
        except Exception as E: out = E
    stop = time.time()
    if out=='': out = 'sm %s processed in %s sec'%(sm,round(stop-start,2))
    return [{sm:out}]

def merge_samples(hdf5_dir,hdf5_out,labels=['DEL','DUP','INV','INS','NOT'],not_prop=1.0):
    hdf5_files = glob.glob(hdf5_dir+'/*.label.hdf5')
    print('merging the following files: %s'%hdf5_files)
    out_f = File(hdf5_out,'a')
    for hdf5_file in hdf5_files:
        print('reading %s'%hdf5_file)
        try:
            in_f = File(hdf5_file,'r')
            for sample in in_f:
                print('reading sample %s'%sample)
                g_id = out_f.require_group('/%s'%sample)
                for k in in_f[sample]:
                    match,sv_type = False,k #will do extact or partial match of labels: NOT=>NOT.1.0 = match!
                    for l in labels:
                        if sv_type.find(l)>-1: match = True
                    if match:
                        if k.find('NOT')>=0 and not_prop<1.0: #::::::::::::::SAMPLING MERGE:::::::::::::::::::::::::::
                            g_path = '/%s/%s'%(sample,k)
                            g_size = (max(1,int(round(not_prop*in_f[sample][k].shape[0]))),)+in_f[sample][k].shape[1:]
                            data = out_f.create_dataset(g_path,g_size,in_f[sample][k].dtype,compression='lzf')
                            idx = np.sort(np.random.choice(range(in_f[sample][k].shape[0]),g_size[0],replace=False))
                            buff = np.zeros(in_f[sample][k].shape,dtype=in_f[sample][k].dtype)
                            buff[:] = in_f[sample][k][:]
                            data[:] = buff[idx][:]
                            for a in in_f[sample][k].attrs:
                                data.attrs[a] = in_f[sample][k].attrs[a]
                        else: #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                            g_path = '/%s/%s'%(sample,k)
                            print('copying %s file\n\tg_path=%s'%(hdf5_file,g_path))
                            in_f.copy(g_path,g_id,name=k)
            in_f.close()
        except Exception as E:
            print(E)
    out_f.close()
    return True

def rename_sample_label(hdf5_in,hdf5_out,sm):
    out_f = File(hdf5_out,'a')
    print('reading %s'%hdf5_in)
    try:
        in_f   = File(hdf5_in,'r')
        sample = sorted(in_f)[0]
        print('renaming sample %s to %s'%(sample,sm))
        g_id = out_f.require_group('/%s'%sm)
        for k in in_f[sample]:
            g_path = '/%s/%s'%(sample,k)
            print('copying file=%s, g_path=%s to file=%s, g_path=%s'%(hdf5_in,g_path,hdf5_out,'/%s/%s'%(sm,k)))
            in_f.copy(g_path,g_id,name=k)
        in_f.close()
    except Exception as E:
        print(E)
    out_f.close()
    return True

#w optionally can take the window size from a normalized zoom type data and select just one window
def rename_sample_norm(hdf5_in,hdf5_out,sm,w=25):
    import h5py
    out_f = File(hdf5_out,'a')
    print('reading %s'%hdf5_in)
    try:
        in_f   = File(hdf5_in,'r')
        sample = sorted(in_f)[0]
        print('renaming sample %s to %s'%(sample,sm))
        for rg in in_f[sample]:
            g_rg_id = out_f.require_group('/%s/%s'%(sm,rg))
            for seq in in_f[sample][rg]:
                if type(in_f[sample][rg][seq]) is h5py._hl.dataset.Dataset: #five style norm
                    g_path = '/%s/%s/%s'%(sample,rg,seq)
                    print('copying %s=>%s to %s=>%s/%s'%(hdf5_in,g_path,hdf5_out,'/%s/%s'%(sm,rg),seq))
                    in_f.copy(g_path,g_rg_id,name=seq)
                else:                                                       #zoom style norm
                    w_key  = str(w)
                    g_path = '/%s/%s/%s/%s'%(sample,rg,seq,w_key)
                    if w_key in in_f[sample][rg][seq]:
                        print('copying %s=>%s/%s to %s=>%s/%s'%(hdf5_in,g_path,w_key,hdf5_out,'/%s/%s'%(sm,rg),seq))
                        in_f.copy(g_path,g_rg_id,name=seq)
        in_f.close()
    except Exception as E:
        print(E)
    out_f.close()
    return True

#some normalization checking for dimension and values => mean should be at 0.0, std should be 1.0...
def check_norms(hdf5_dir,mean=0.0,std=1.0):
    hdf5s = glob.glob(hdf5_dir+'*.norm.*')
    R = {}
    for h5 in hdf5s:
        f = File(h5,'r')
        for sm in f:
            for rg in f[sm]:
                for seq in f[sm][rg]:
                    attrs = f[sm][rg][seq].attrs
                    if 'l' in attrs and 'm' in attrs and 'feature_order' in attrs:
                        dims = f[sm][rg][seq].shape
                        lns  = f[sm][rg][seq].attrs['l']
                        ftrs = f[sm][rg][seq].attrs['feature_order']
                        mns  = f[sm][rg][seq].attrs['m']
                        if dims!=(lns,len(ftrs),mns):
                            print('attributes are mismatched to shape on: sm=%s,rg=%s,seq=%s'%(sm,rg,seq))
                        else:
                            if dims[1:] not in R: R[dims[1:]] = {sm:{rg:seq}}
                            else:                 R[dims[1:]][sm] = {rg:seq}
                    else:
                        for w in f[sm][rg][seq]:
                            dims = f[sm][rg][seq][w].shape
                            lns  = f[sm][rg][seq][w].attrs['l']
                            ftrs = f[sm][rg][seq][w].attrs['feature_order']
                            mns  = f[sm][rg][seq][w].attrs['m']
                            if dims!=(lns,len(ftrs),mns):
                                print('attributes are mismatched to shape on: sm=%s,rg=%s,seq=%s,w=%s'%(sm,rg,seq,w))
                            else:
                                if dims[1:] not in R: R[dims[1:]] = {sm:{rg:{seq:w}}}
                                else:                 R[dims[1:]][sm] = {rg:{seq:w}}
    if len(R)>1:
        print('presence of non-uniform normalization files detected...')
        print(R)
    return True

def check_labels(hdf5_dir):
    return True

maxima_result_list = []
def collect_maxima_results(result):
    maxima_result_list.append(result)

label_result_list = []
def collect_label_results(result):
    label_result_list.append(result)

if __name__ == '__main__':
    des = """Data Prep: VCF Mining [+] HFM to Tensor Normalization/Targeting Tool v0.1.7\nCopyright (C) 2020-2021 Timothy James Becker\n"""
    parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--vcf_in_path',type=str,help='g1kp3 new genotyped all-sample vcf file\t[None]')
    parser.add_argument('--hfm_in_path',type=str,help='full hfm.hdf5 file directory\t[None]')
    parser.add_argument('--out_path',type=str,help='targets,tensors,models,plots output path\t[None]')
    parser.add_argument('--types',type=str,help='comma seperated svtypes\t[DEL,DUP,INV,INS,NOT]')
    parser.add_argument('--sub_types', action='store_true', help='make use of somacx subtypes: DUP:DISPERSED, etc\t[False]')
    parser.add_argument('--no_standard',action='store_true',help='do not use mean center, with standard deviation scaling\t[False]')
    parser.add_argument('--range',type=str,help='comma seperated float range to scale to\t[None]')
    parser.add_argument('--n_w',type=int,help='number of flanking windows per side for targeting\t[5]')
    parser.add_argument('--flank_w',type=int,help='minimum flank size for background mask used in targeting phase\t[100]')
    parser.add_argument('--shift_w',type=int,help='number of windows left and right to shift in targeting phase\t[2]')
    parser.add_argument('--mean_cov',type=int,help='mean coverage to use for calculating maxima regions\t[20]')
    parser.add_argument('--min_size',type=int,help='minimum size bp for labels to span\t[25]')
    parser.add_argument('--not_prop',type=float,help='proportion of background regions to include per sample\t[1.0]')
    parser.add_argument('--cpus',type=int,help='number of cpus to use\t[1]')
    parser.add_argument('--comp',type=str,help='gzip(slow) or lzf(fast) data compresison\t[lzf]')
    parser.add_argument('--zoom', action='store_true', help='use zoom instead of five point method\t[False]')
    parser.add_argument('--verbose', action='store_true', help='include more information\t[False]')
    parser.add_argument('--hgsv_vcf',action='store_true',help='use hgsv illumina vcf format\t[False]')
    parser.add_argument('--fusorsv_vcf',action='store_true',help='use fusorsv vcf format\t[False]')
    parser.add_argument('--somacx_vcf',action='store_true',help='use extended somacx vcf format\t[False]')
    parser.add_argument('--clean_labels',action='store_true',help='clean up intermediate files\t[False]')
    args = parser.parse_args()
    if args.types is not None:       types    = args.types.split(',')
    else:                            types    = ['DEL','DUP','INV','INS','NOT']
    if args.range is not None:       rng      = [float(r) for r in args.range.split(',')]
    else:                            rng      = None
    if args.n_w is not None:         n_w      = args.n_w
    else:                            n_w      = 5
    if args.mean_cov is not None:    mean_cov = args.mean_cov
    else:                            mean_cov = 45
    if args.not_prop is not None:    not_prop = args.not_prop
    else:                            not_prop = 1.0
    if args.cpus is not None:        cpus     = args.cpus
    else:                            cpus     = 1
    if args.comp is not None:        comp     = args.comp
    else:                            comp     = 'lzf'
    if args.vcf_in_path is not None: vcf_in_path = args.vcf_in_path #human_g1k_v37_decoy.all.geno.vcf.gz'
    else:                            vcf_in_path = ''
    if args.out_path is not None:    out_path = args.out_path
    else:                            raise IOError
    if args.flank_w is not None :    flank_w  = args.flank_w
    else:                            flank_w  = 100
    if args.flank_w is not None :    shift_w  = args.shift_w
    else:                            shift_w  = 2
    if args.min_size is not None :   min_size = args.min_size
    else:                            min_size = 25
    if args.zoom:                    zoom     = True
    else:                            zoom     = False
    if args.verbose:                 verbose  = True
    else:                            verbose  = False
    target_path = out_path+'/targets/'
    tensor_path = out_path+'/tensors/'
    if not os.path.exists(target_path): os.mkdir(target_path)
    if not os.path.exists(tensor_path): os.mkdir(tensor_path)
    if args.hfm_in_path is None: hfm_files = []
    else:                        hfm_files = sorted(glob.glob(args.hfm_in_path+'/*.hdf5'))
    if len(hfm_files)<1:
        hfm_files = sorted(glob.glob(tensor_path+'*.norm.hdf5'))
    hfm_sms   = [hfm_file.rsplit('/')[-1].rsplit('.')[0] for hfm_file in hfm_files]
    vcf_files = sorted(glob.glob(vcf_in_path+'/*.vcf*'))
    vcf_sms   = [vcf_file.rsplit('/')[-1].rsplit('.')[0] for vcf_file in vcf_files]
    if len(vcf_sms)>0: sms = sorted(list(set(hfm_sms).intersection(set(vcf_sms))))
    else:              sms = sorted(hfm_sms)
    print('hfm_sms=%s' % hfm_sms)
    print('vcf_sms=%s' % vcf_sms)
    print('sms=%s'%sms)
    if len(hfm_sms)>0 and all([os.path.exists(target_path+'/%s.pickle.gz'%sm) for sm in hfm_sms]):
        print('located target files for all hfm samples')
    if len(vcf_sms)>0 and all([os.path.exists(target_path+'/%s.pickle.gz'%sm) for sm in vcf_sms]):
        print('located target files for all vcf samples')
    if len(sms)>0 and all([os.path.exists(target_path+'/%s.pickle.gz'%sm) for sm in sms]):
        print('located target files for all hfm/vcf samples')
    else:
        print('preparing target files from maxima, ref and vcf input files...')
        vcf_in_dir = '/'.join(vcf_in_path.rsplit('/')[:-1])
        if not os.path.exists(vcf_in_dir+'/sms.maxima.json'):
            m_start = time.time()
            p1 = mp.Pool(processes=cpus)
            for hfm_file in hfm_files:
                print('getting maxima sm=%s'%hfm_file)
                p1.apply_async(get_maxima_regions,
                               args=(hfm_file,out_path,mean_cov/10.0,mean_cov*10.0,True),
                               callback=collect_maxima_results)
                time.sleep(0.5)
            p1.close()
            p1.join()
            R = {}
            for result in maxima_result_list: R[list(result.keys())[0]] = result[list(result.keys())[0]]
            I = intersect_maxima_regions_ooc(out_path,min_w=int(2*flank_w)) #write out the maxima regions
            m_stop = time.time()
            print('completed maxima regions in %s sec'%round(m_stop-m_start,2))
            #get_maxima_regions(hfm_files,lower_cut=mean_cov/10.0,upper_cut=10.0*mean_cov)
        with open(vcf_in_dir+'/sms.maxima.json','r') as f: mask = json.load(f)
        with open(vcf_in_dir+'/ref.meta.json','r') as f:   seqs = json.load(f)
        # D = hgsv_illumina_vcf_to_dict(vcf_path=vcf_in_path,out_path=vcf_in_dir)
        if args.somacx_vcf:
            print('using extended somacx vcf format conversions')
            D = somacx_vcf_to_dict(vcf_dir=vcf_in_path,out_path=vcf_in_dir,
                                   types=types,sub_types=args.sub_types,min_size=min_size)
        elif args.fusorsv_vcf:
            print('using fusorsv vcf format conversions')
            D = fusorsv_vcf_to_dict(vcf_in_path,out_path=vcf_in_dir,samples=hfm_sms)
        elif args.hgsv_vcf:
            min_call,pac_bio,hybrid = 1,True,True
            print('using hgsv illumina vcf format conversions')
            D = hgsv_illumina_vcf_to_dict(vcf_in_path,out_path=vcf_in_dir,samples='all',min_call=min_call,pac_bio=pac_bio,hybrid=hybrid,verbose=False)
            print(':::distributions given parames=[min_call=%s, pac_bio=%s, hybrid=%s]'%(min_call,pac_bio,hybrid))
            print(':::DEL=%s'%([len(D['sample'][sm]['DEL']) for sm in D['sample']]))
            print(':::DUP=%s'%([len(D['sample'][sm]['DUP']) for sm in D['sample']]))
            print(':::INV=%s'%([len(D['sample'][sm]['INV']) for sm in D['sample']]))
        else: #g1kp3 is the default
            print('using default g1kp3 vcf format conversions')
            D = g1kp3_vcf_to_dict(vcf_path=vcf_in_path,out_path=vcf_in_dir,samples=hfm_sms)
        Q,R,M = sv_regions_by_type(D,mask,seqs,flank=flank_w,somatic=args.somacx_vcf) #M is corrected calls, R is the nonsv=>NOT, Q is the sv
        write_sv_dict(M,R,target_path,min_size=min_size,somatic=args.somacx_vcf,summarize=True)
    time.sleep(5.0) # take a quick break...
    #||----------------------------------------------------------------------
    print('starting normalization and targeting phase..')
    time.sleep(5.0)
    t_start = time.time()
    p2 = mp.Pool(processes=cpus)
    for hfm_in in sorted(hfm_files):
        print('hfm_file=%s'%hfm_in)
        sm           = hfm_in.rsplit('/')[-1].rsplit('.')[0]
        target_in    = target_path+'/%s.pickle.gz'%sm
        hdf5_out    = tensor_path+'/%s.norm.hdf5'%sm
        hdf5_label   = tensor_path+'/%s.label.hdf5'%sm
        if not os.path.exists(hdf5_label):
            print('queuing sample sm=%s'%sm)
            p2.apply_async(process_labels,
                           args=(sm,hfm_in,hdf5_out,target_in,hdf5_label,types,not_prop,n_w,
                                 (not args.no_standard),rng,comp,zoom,shift_w,verbose),
                           callback=collect_label_results)
        else:
            print('sample sm=%s was already prepped'%sm)
        time.sleep(0.5)
    p2.close()
    p2.join()
    for result in label_result_list: print(result)
    if args.clean_labels and not os.path.exists(tensor_path+'/all.label.hdf5'):
        merge_samples(tensor_path,tensor_path+'/all.label.hdf5')
        labels = glob.glob(tensor_path+'/*.label.hdf5')
        labels = list(set(labels).difference(set([(tensor_path+'/all.label.hdf5').replace('//','/')])))
        print('cleaning single label target tensors: %s'%labels)
        for label in labels: os.remove(label)
    t_stop = time.time()
    print('total time was %s sec'%round(t_stop-t_start,2))
