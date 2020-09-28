# RMKMCplus

This is a reimplementation of the RMKMC algorithm. The RMKMC algorithm is proposed by Cai, Nie, Huang in their paper "Multi-view K-Means Clustering on Big Data (2013)". You can find the original paper at http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.415.8610&rep=rep1&type=pdf.

The significance of this project is that it embeds an intialization step natually into the optimization steps of the original algorithm. It's a small change and a concise description can be found in "Initialization.pdf".

This is originally the final group project of Spring 2020 semester's COSI 126A at Brandeis University. Students were asked to reimplement some algorithm they read from papers and make some changes to boost the performance. I worked with Cheng Chen, Alex Chang and Honey Park on it. I alone designed the initialization step and found it rather inspiring, which is why I decided to post it online.

The algorithm with the added initialization step is in 'RMKMC.py'. It is tested on the UCI handwritten numerals dataset (size 2000) which can be found at https://archive.ics.uci.edu/ml/datasets/Multiple+Features. 
