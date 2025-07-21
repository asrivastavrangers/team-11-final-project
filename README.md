# team-11-final-project
AAI-511: Deep Learning and Neural Networks

# Music Genre and Composer Classification Using Deep Learning

## Project Overview

This project focuses on **composer classification** using deep learning techniques. Classical music, with its structured compositions and rich textures, provides an ideal domain to explore how artificial intelligence can learn to distinguish between styles of renowned composers. We aim to classify compositions from **Bach**, **Beethoven**, **Chopin**, and **Mozart** using two deep learning models: **LSTM** and **CNN**.

---

## Objective

The **primary objective** is to accurately identify the composer of a given musical score in MIDI format.

To achieve this, we:

* Preprocess and extract musical features from MIDI files
* Build and compare **LSTM** and **CNN** models
* Optimize the models for improved performance
* Evaluate classification accuracy using appropriate metrics

---

## Dataset Description

* **Source**: [Kaggle - MIDI Classic Music Dataset](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)
* **Total Files**: 3,929 MIDI files
* **Selected Composers**: Johann Sebastian Bach, Ludwig van Beethoven, Frédéric Chopin, and Wolfgang Amadeus Mozart
* **Dataset Size**: \~500 MB
* **Metadata Includes**:

  * File name
  * Composer label
  * Composition details

The dataset contains musical scores in MIDI format, allowing us to extract:

* **Pitch**
* **Note duration**
* **Tempo**
* **Time signature**
* **Chord progressions**

These features are crucial for training deep learning models.

---

## Team Members & Contributions

### Abhay Srivastav

* Data Collection & Filtering
* Data Preprocessing
* Feature Engineering
* Data Visualization
* Documentation

### Balubhai Sukani

* Model Building: LSTM and CNN
* Model Training & Evaluation
* Model Optimization

### Aditya Sourabh

* Technical Report
* Notebook Finalization
* Model Performance Review & Improvement

---

## Tech Stack

* Python (NumPy, Pandas, Matplotlib, Seaborn)
* Deep Learning: TensorFlow / Keras
* MIDI Processing: `pretty_midi`, `music21`
* Data: KaggleHub, MIDI files
* IDEs: Jupyter Notebook, VS Code
