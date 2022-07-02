# ECG signals (1000 fragments)

Link to the database : https://data.mendeley.com/datasets/7dybx7wyfn/3
Link to one article : https://www.sciencedirect.com/science/article/pii/S0957417417306292?via%3Dihub

# Description

For research purposes, the ECG signals were obtained from the PhysioNet service (http://www.physionet.org) 
from the MIT-BIH Arrhythmia database. The created database with ECG signals is described below.

1) The ECG signals were from 45 patients: 19 female (age: 23-89) and 26 male (age: 32-89). 
2) The ECG signals contained 17 classes: normal sinus rhythm, pacemaker rhythm, and 15 types of cardiac dysfunctions (for each of which at least 10 signal fragments were collected). 
3) All ECG signals were recorded at a sampling frequency of 360 [Hz] and a gain of 200 [adu / mV]. 
4) For the analysis, 1000, 10-second (3600 samples) fragments of the ECG signal (not overlapping) were randomly selected. 
5) Only signals derived from one lead, the MLII, were used. 
6) Data are in mat format (Matlab).


## Normal data

"1 NSR" : normal sinus rhythm (283)
"17 PR" : pacemaker rhythm (45)

## cardiac dysfunctions

"Name of the data folder" : name of the dysfunction (number of data)

"2 APB" : Atrial premature beat (66)
"3 AFL" : Atrial flutter (20)
"4 AFIB" : Atrial fibrillation (135)
"5 SVTA" : Supraventricular tachyarrhythmia (13)
"6 WPW" : Wolff-Parkinson-White syndrome (21)
"7 PVC" : Premature ventricular contraction (133)
"8 Bigeminy" : Ventricular bigeminy (55)
"9 Trigeminy" : Ventricular trigeminy (13)
"10 VT" : Ventricular tachycardia (10)
"11 IVR" : Idioventricular rhythm (10)
"12 VFL" : Ventricular flutter (10)
"13 Fusion" : Fusion of ventricular and normal beat (11)
"14 LBBBB" : Left bundle branch block beat (103)
"15 RBBBB" : Right bundle branch block beat (62)
"16 SDHB" : Second-degree heart block (10)


