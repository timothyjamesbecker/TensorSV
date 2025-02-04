![GitHub All Releases](https://img.shields.io/github/downloads/timothyjamesbecker/TensorSV/total.svg) 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)<br>
![Alt text](images/tensorsv_logo.png?raw=true "not_tensor") <br>
## Moment Based SV Calling and Genotyping
Copyright (C) 2020-2021 Timothy James Becker
```bash
T. Becker and D.G. Shin,"TensorSV: structural variation inference 
using tensors and variable topology neural networks", 
2020 IEEE BIBM, Seoul, Korea (South), 2020, pp. 1356-1360
```
## Requirements:
<b>python 3.6+</b><br>
cython 0.29+<br>
numpy 1.18+<br>
matplotlib 3.2.1<br>
h5py 2.10<br>
pysam 0.15.2<br>
hfm 0.1.8<br>
tensorflow 1.15.0 (works with tensorflow-gpu 1.15.0 for GPU as well)<br>
## PIP Installation:
```bash
python3 -m pip install https://github.com/timothyjamesbecker/TensorSV/releases/download/0.0.1/tensorsv-0.0.1.tar.gz
```
## Basic Usage:
<b>(1)</b> Start by extracting the features from the BAM file using the hfm package. The script being used here: <b>extractor.py</b> is a high_level multi-bam aware extraction runner that ships with the hfm package.  You can install this package from the git repo: https://github.com/timothyjamesbecker/hfm
```bash
extractor.py \
--ref_path ./reference_sequence.fa \
--in_path ./folder_of_bam_files/ \
--out_dir ./output_hdf5_files/ \
--seqs chr1,chr2, ... ,chr22,chrX,chrY,chrM \
--window 25 \
--branch 2 \
--cpus 12
```
<b>(2)</b> Next you need to normalize and standardize the HFM hdf5 files and capture targets if training is desired using the TensorSV script <b>data_prep.py</b> shown below.  This script can run in parallel for each sample so setting your cpus to the number of samples when you have enough processors and memory is suggested. The result of this step will produce one *.norm.hdf file and one *.label.hdf per sample. For training you can run the data_prep.merge_samples function to mix together any samples that have under gone this process.
```bash
data_prep.py \
--vcf_in_path ./hgsv_hg38_hfm_server/hgsv.illumina.hg38.all.geno.vcf.gz \
--hfm_in_path ./hgsv_hg38_hfm_server/ \
--out_path ./hgsv_hg38_hfm_server/ \
--cpus 9
```
<b>(3)</b> Now you can either train a new SV model using <b>train_sv.py</b> or use an existing one in the next step.
```bash
train_sv.py \
--in_path ./hgsv_hg38_hfm_server/tensors/hgsv.hg38.labels.hdf5 \
--sub_sample_not 0.75 \
--out_path ./hgsv_hg38_hfm_server/cnn_75/ \
--sv_types DEL \ 
--filters all \ 
--form cnn  \ 
--levels 2,4 \
--cmxs 2,4  \
--batches 32,64,128 \
--epochs 10,25 \
--decays 1e-5,2e-5 \
--split_seed 0 \
--gpu_num 0 \
--verbose
```
<b>(4)</b> Now you can run the <b>predict_sv.py</b> on the normalized hdf5 from step <b>(2)</b> If you have used training diagnostics and your folder contains true and comparable calls, this will produce metrics on your calls to show your model accuracy.
```bash
predict_sv.py \
--base_dir ./hgsv_hg38_hfm_server/ \
--run_dir ./hgsv_hg38_hfm_server/cnn_75/ \
--out_dir ./hgsv_hg38_hfm_server/cnn_75_result/ \
--samples  HG00096,HG00268 \
--sv_type DEL \
--seqs chr19,chr20,chr21,chr22 \
--gpu_num 0
```
