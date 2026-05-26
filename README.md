# ECG Detection for YODA (Internal Use)

This section documents how to run ECGFounder inference on YODA PSG data. It produces per-window (10-second) probabilities for 150 ECG classes including atrial fibrillation, bundle branch block, infarct patterns, arrhythmias, and more.

## Setup

**1. Install dependencies**

```bash
conda activate ECGFounder   # or your env with requirements.txt installed
pip install -r requirements.txt
```

**2. Download the checkpoint** (one-time, if not already present)

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('PKUDigitalHealth/ECGFounder', '1_lead_ECGFounder.pth', local_dir='./checkpoint')
"
```

The checkpoint will be saved to `./checkpoint/1_lead_ECGFounder.pth`.

## Running Inference

Edit `infer_h5.sh` to set your input/output paths, then run:

```bash
bash infer_h5.sh
```

Key variables in `infer_h5.sh`:

| Variable | Description | Default |
|---|---|---|
| `H5_DIR` | Directory containing input H5 files | `/your/input/h5_dir/` |
| `OUT` | Output directory for result CSVs | `/your/output/results_dir/` |
| `LEADS` | ECG channel name under `/signals/` in the H5 file | `ecg` |
| `WINDOW_SEC` | Sliding window length in seconds | `10` |
| `STRIDE_SEC` | Stride between windows (10 = no overlap) | `10` |
| `REWRITE` | `yes` = reprocess all; `no` = skip already-done files | `yes` |

## H5 Input Format

The script supports two H5 layouts:

**Grouped format** (default, used for YODA) — H5 files produced by `cohort_h5_conversion_reference.py`:
```
<file>.h5
  attrs: sampling_rate
  /signals/ecg    shape: (T,) or (T, 1)   float, in V or µV
```
Use `--leads ecg` (already set in `infer_h5.sh`).

**Legacy Xy matrix format** (old CAISR/Morgoth2 PSG files):
```
<file>.h5
  /X    shape: (n_channels, n_samples)
```
Pass `--xy_channel_idx 13 --xy_fs 200` instead of `--leads`.

## Output

One CSV per H5 file is written to `OUT/`, named `<h5_stem>.csv`:

```
file, window_start_sec, window_end_sec, ABNORMAL ECG, NORMAL SINUS RHYTHM, ATRIAL FIBRILLATION, ...
```

Each row is one 10-second window. Values are sigmoid probabilities (0–1) for each of the 150 classes. The full class list is in `tasks.txt`.

## 150 ECG Classes

Classes marked with **★** are highlighted as important and most relevant to sleep research.

**Normal / General ECG Labels** (1–6)
1. Normal ECG
2. Abnormal ECG
3. ★ Borderline ECG
4. ★ Otherwise normal ECG
5. ★ Sinus rhythm
6. Normal sinus rhythm

**Sinus Rhythm / Heart Rate Disorders** (7–23)
7. Sinus bradycardia
8. ★ Marked sinus bradycardia
9. Sinus tachycardia
10. ★ With sinus arrhythmia
11. With marked sinus arrhythmia
12. Junctional bradycardia
13. Junctional rhythm
14. Idioventricular rhythm
15. Ectopic atrial rhythm
16. Undetermined rhythm
17. Multifocal atrial tachycardia
18. Supraventricular tachycardia
19. Ventricular tachycardia
20. Wide QRS tachycardia
21. Wide QRS rhythm
22. With sinus pause
23. With undetermined rhythm irregularity

**Atrial Arrhythmias** (24–27)
24. Atrial fibrillation
25. With rapid ventricular response
26. With slow ventricular response
27. Atrial flutter

**Premature / Ectopic Beats** (28–38)
28. Premature atrial complexes
29. Premature supraventricular complexes
30. Premature ventricular complexes
31. Premature ectopic complexes
32. Premature ventricular and fusion complexes
33. Fusion complexes
34. Supraventricular complexes
35. With premature ventricular or aberrantly conducted complexes
36. With ventricular escape complexes
37. With junctional escape complexes
38. In a pattern of bigeminy

**AV Block / SA Block / Conduction Disorders** (39–49)
39. With 1st degree AV block
40. With prolonged AV conduction
41. With complete heart block
42. With variable AV block
43. With 2nd degree AV block Mobitz I
44. With 2nd degree SA block Mobitz I
45. With 2nd degree SA block Mobitz II
46. With 2:1 AV conduction
47. With AV dissociation
48. With retrograde conduction
49. With short PR

**Bundle Branch / Fascicular Blocks** (50–65)
50. Right bundle branch block
51. Incomplete right bundle branch block
52. Left bundle branch block
53. Incomplete left bundle branch block
54. ★ Left anterior fascicular block
55. ★ Left posterior fascicular block
56. ★ Bifascicular block
57. ★ RBBB and left anterior fascicular block
58. ★ RBBB and left posterior fascicular block
59. Nonspecific intraventricular conduction delay
60. Nonspecific intraventricular block
61. ★ RSR' pattern in V1
62. ★ RSR' or QR pattern in V1 suggests right ventricular conduction delay
63. ★ Blocked
64. ★ Masked by fascicular block
65. Aberrant conduction

**Axis Abnormalities** (66–72) ★
66. ★ Left axis deviation
67. ★ Abnormal left axis deviation
68. ★ Leftward axis
69. ★ Right axis deviation
70. ★ Abnormal right axis deviation
71. ★ Rightward axis
72. ★ Right superior axis deviation

**Chamber Enlargement / Hypertrophy** (73–80)
73. Left atrial enlargement
74. Right atrial enlargement
75. Biatrial enlargement
76. ★ Left ventricular hypertrophy
77. ★ Voltage criteria for left ventricular hypertrophy
78. ★ Right ventricular hypertrophy
79. ★ Biventricular hypertrophy
80. Pulmonary disease pattern

**Myocardial Infarction / Injury / Ischemia** (81–97) ★
81. ★ Anterior infarct
82. ★ Anteroseptal infarct
83. ★ Anterolateral infarct
84. ★ Septal infarct
85. ★ Lateral infarct
86. ★ Inferior infarct
87. ★ Inferior-posterior infarct
88. ★ Posterior infarct
89. ★ Acute MI
90. ★ Acute MI / STEMI
91. ★ Acute
92. ★ Anterior injury pattern
93. ★ Anterolateral injury pattern
94. ★ Inferior injury pattern
95. ★ Inferolateral injury pattern
96. ★ Lateral injury pattern
97. ★ Consider right ventricular involvement in acute inferior infarct

**ST Segment Abnormalities** (98–109)
98. Nonspecific ST abnormality
99. Nonspecific ST and T wave abnormality
100. ★ Non-specific change in ST segment in
101. ★ ST now depressed in
102. ★ ST no longer depressed in
103. ★ ST more depressed in
104. ★ ST less depressed in
105. ★ ST elevation now present in
106. ★ ST elevation has replaced ST depression in
107. ★ ST no longer elevated in
108. ★ ST more elevated in
109. ★ ST less elevated in

**T Wave Abnormalities** (110–120)
110. Nonspecific T wave abnormality
111. ★ Nonspecific T wave abnormality now evident in
112. ★ Nonspecific T wave abnormality no longer evident in
113. ★ T wave inversion now evident in
114. ★ T wave inversion no longer evident in
115. ★ T wave inversion more evident in
116. ★ T wave inversion less evident in
117. ★ Inverted T waves have replaced nonspecific T wave abnormality in
118. ★ Nonspecific T wave abnormality has replaced inverted T waves in
119. ★ T wave amplitude has decreased in
120. ★ T wave amplitude has increased in

**QT / QRS Abnormalities** (121–127) ★
121. ★ Low voltage QRS
122. ★ With QRS widening
123. ★ With QRS widening and repolarization abnormality
124. ★ With repolarization abnormality
125. ★ Prolonged QT
126. ★ QT has lengthened
127. ★ QT has shortened

**Pacemaker / Pacing Related** (128–142)
128. Electronic atrial pacemaker
129. Electronic ventricular pacemaker
130. AV sequential or dual chamber electronic pacemaker
131. AV dual-paced rhythm
132. AV dual-paced complexes
133. Atrial-paced rhythm
134. Atrial-paced complexes
135. Ventricular-paced rhythm
136. Ventricular-paced complexes
137. Atrial-sensed ventricular-paced rhythm
138. Biventricular pacemaker detected
139. Electronic demand pacing
140. ★ Suspect unspecified pacemaker failure
141. Sinus/atrial capture
142. With a competing junctional pacemaker

**Other Specific Diagnoses** (143–150)
143. Wolff-Parkinson-White
144. Early repolarization
145. ★ Acute pericarditis
146. No P-waves found
147. Or digitalis effect
148. ★ Pediatric ECG analysis
149. ★ R in AVL
150. ★ Anterolateral leads

---

# ECGFounder: An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains

This is the official implementation of our paper "[An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains](https://arxiv.org/abs/2410.04133)".

> Authors: Jun Li, Aaron Aguirre, Junior Moura, Jiarui Jin, Che Liu, Lanhai Zhong, Chenxi Sun, Gari Clifford, Brandon Westover, Shenda Hong.

Try online demo at http://ai.heartvoice.com.cn/diagnosis.html

## 🚀 Getting Started

🚩 **News** 
(Aug 2025): The out-of-the-box feature — the 150-class classification validation function of ECGFounder — is now online.

(Mar 2025): The pre-training checkpoint is now available on [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main)!


> ⚠️ **Important Notice**  
> If you intend to use the pretrained model weights for validation or fine-tuning, you must **strictly** follow the preprocessing steps in **dataset.py** — including **filtering**, **z-score normalization**, and any other specified procedures.  
> Failure to do so will make it difficult to reproduce the results reported in the paper!


### Installation

To clone this repository:

```
git clone https://github.com/PKUDigitalHealth/ECGFounder.git
```

### Environment Set Up

Install required packages:

```
conda create -n ECGFounder python=3.10
conda activate ECGFounder
pip install -r requirements.txt
```

### 150-class classification validation

* **PTB-XL**: Please download the [PTB-XL](https://www.physionet.org/content/ptb-xl/1.0.3/) dataset from physionet.

Next, please download the model's checkpoint from the  [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main). And place the model weights in path *./checkpoint*

You can run the *ptbxl_eval.py* to do the 150-class classification validation on PTB-XL dataset.


### Fine-tune on Downstream Tasks

In our paper, downstream datasets we used are as follows:

* **MIMIC-ECG**: Please download the [MIMIC-ECG](https://physionet.org/content/mimiciv/2.2/) dataset from physionet.


You can run the jupyter notebook to finetune the model by the example dataset.



## References

If you found our work useful in your research, please consider citing our works at:
> ```
> @article{li2025electrocardiogram,
>   title={An Electrocardiogram Foundation Model Built on over 10 Million Recordings},
>   author={Li, Jun and Aguirre, Aaron D and Junior, Valdery Moura and Jin, Jiarui and Liu, Che and Zhong, Lanhai and Sun, Chenxi and Clifford, Gari and Brandon Westover, M and Hong, Shenda},
>   journal={NEJM AI},
>   volume={2},
>   number={7},
>   pages={AIoa2401033},
>   year={2025},
>   publisher={Massachusetts Medical Society}
> }
> ```
