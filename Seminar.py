#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:06:09 2023

@author: aaryanjha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
import sys
import os
import pyaudio
import wave
import tempfile
import librosa
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Bidirectional, Flatten
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QProgressBar, QWidget, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import QEventLoop
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QFont
from PyQt5 import sip
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
from PyQt5.QtCore import QSize


###DATA COLLECTION
def readTess(tess_directory = '/Users/aaryanjha/Desktop/Comp_Sci_Seminar/TESS_Toronto_emotional_speech_set_data/', resample_sr=8000):
    emotions = {'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad', 'angry': 'angry', 'fear': 'fear', 'disgust': 'disgust', 'ps': 'surprise'}
    file_emotion = []
    file_path = []   
    ds_store_file_location = os.path.join(tess_directory, '.DS_store')
    if os.path.isfile(ds_store_file_location):
        os.remove(ds_store_file_location)
        
    for dir in os.listdir(tess_directory):
        for file in os.listdir(os.path.join(tess_directory, dir)):
            if  file != '.DS_Store' and file.endswith('.wav'):
                part = file.split('.')[0]
                emotion_label = part.split('_')[-1]
                if emotion_label in emotions:
                    file_emotion.append(emotions[emotion_label])
                    file_path.append(os.path.join(tess_directory, dir, file))     
                    
    df = pd.DataFrame()
    df['speech'] = file_path
    df['label'] = file_emotion
    df['source'] = 'TESS'
    return df

def readRav(resample_sr=8000, rav_directory='/Users/aaryanjha/Desktop/Comp_Sci_Seminar/Ravdess/dataset'):
    file_emotion = []
    file_path = []
    for dirname, _, filenames in os.walk(rav_directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path.append(os.path.join(dirname, filename))
                label = int(filename.split('-')[2])
                file_emotion.append(label)

    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
    emotions_list = [emotions[label] for label in file_emotion]

    df = pd.DataFrame()
    df['speech'] = file_path
    df['label'] = emotions_list
    df['source'] = 'RAV'
    return df

# Combine the TESS and RAVDESS datasets and reset the index
def combine():
    df_tess = readTess().reset_index(drop=True)
    df_rav = readRav().reset_index(drop=True)
    df_combined = pd.concat([df_tess, df_rav], axis=0).reset_index(drop=True)
    return df_combined


###DATA PREPROCESS

def preprocess(resample_sr=8000):
    df = combine()    
    df['audio'] = df['speech'].apply(lambda x: librosa.load(x, sr=resample_sr)[0])  
    df['zcr'] = df['audio'].apply(lambda x: np.mean(librosa.feature.zero_crossing_rate(y=x).T, axis=0))
    df['mfcc'] = df['audio'].apply(lambda x: np.mean(librosa.feature.mfcc(y=x, sr=resample_sr, n_mfcc=40).T, axis=0))
    df['features'] = df.apply(lambda row: np.hstack((row['zcr'], row['mfcc'])), axis=1)
    df=df.fillna(0)
    features = pd.DataFrame(df['features'].values.tolist())    
    df = pd.concat([df,features],axis=1)
    df.to_csv("data_path.csv", index = False)
    df=df.fillna(0)  
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['speech','label','source','audio', 'zcr', 'mfcc', 'features'],axis=1), 
                                                        df.label, 
                                                        test_size=0.25, 
                                                        shuffle=True, 
                                                        random_state=42)    
    # One-hot encode the target labels
    encoder = OneHotEncoder()    
    y_train_onehot = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
    y_test_onehot = encoder.transform(np.array(y_test).reshape(-1,1)).toarray()        
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, encoder

###MODEL CREATION AND TRAINING
    
 # Create a Convolutional Neural Network (CNN) Long Short-Term Memory (LSTM) model.       
def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_cnn_lstm_model(X_train, X_test, y_train_onehot, y_test_onehot):
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train_onehot.shape[1]
    model = create_cnn_lstm_model(input_shape, num_classes)
    history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=50, batch_size=32)
    test_accuracy = history.history['val_accuracy'][-1]
    return model, test_accuracy, history

# Train a Support Vector Machine (SVM) model.
def train_svm_model(X_train, X_test, y_train, y_test, encoder):
    svm_model = SVC(kernel='linear', C=1, random_state=42)
    y_train_inv = encoder.inverse_transform(y_train)
    y_test_inv = encoder.inverse_transform(y_test)
    svm_model.fit(X_train, y_train_inv)
    y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)
    train_accuracy = accuracy_score(y_train_inv, y_train_pred)
    test_accuracy = accuracy_score(y_test_inv, y_test_pred)
    return svm_model, test_accuracy

# Train a Convolutional Neural Network (CNN) Bi-directional Long Short-Term Memory (Bi-LSTM) model.
def create_cnn_bidirectional_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_cnn_bidirectional_lstm_model(X_train, X_test, y_train_onehot, y_test_onehot):
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train_onehot.shape[1]
    model = create_cnn_bidirectional_lstm_model(input_shape, num_classes)
    history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=50, batch_size=32)
    test_accuracy = history.history['val_accuracy'][-1]
    return model, test_accuracy, history

#Create an Artificial Neural Network
def create_ann_model(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_ann_model(X_train, X_test, y_train_onehot, y_test_onehot):
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train_onehot.shape[1]
    model = create_ann_model(input_shape, num_classes)
    history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=50, batch_size=32)
    test_accuracy = history.history['val_accuracy'][-1]
    return model, test_accuracy, history


###DATA VISUALIZATION

# Create a waveplot for an audio sample
def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 4))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    duration = len(data) / sr
    time = np.linspace(0, duration, len(data))
    plt.plot(time, data, linewidth=0.5, alpha=0.7, color='black')
    plt.xlabel('Time (s)', size=12)
    plt.ylabel('Amplitude', size=12)
    plt.show()

# Create a spectrogram for an audio sample
def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    plt.colorbar()    

def display_emotion(emotion):
    df = combine()
    path = np.array(df['speech'][df['label']==emotion])[1]
    data, sampling_rate = librosa.load(path)
    create_waveplot(data, sampling_rate, emotion)
    create_spectrogram(data, sampling_rate, emotion)
#Compare the test accuracies between the different models
def plot_test_accuracies(cnn_lstm_history, svm_test_accuracy, cnn_bidirectional_lstm_history, ann_history):
    epochs = len(cnn_lstm_history.history['val_accuracy'])
    svm_test_accuracy_list = [svm_test_accuracy] * epochs
    
    cnn_lstm_accuracies = cnn_lstm_history.history['val_accuracy']
    cnn_bidirectional_lstm_accuracies = cnn_bidirectional_lstm_history.history['val_accuracy']
    ann_accuracies = ann_history.history['val_accuracy']
    
    assert all(0 <= acc <= 1 for acc in cnn_lstm_accuracies + cnn_bidirectional_lstm_accuracies + ann_accuracies + [svm_test_accuracy])

    plt.plot(range(1, epochs+1), cnn_lstm_accuracies, label='CNN-LSTM', color = 'blue')
    plt.plot(range(1, epochs+1), svm_test_accuracy_list, label='SVM', color='r', linestyle='--')
    plt.plot(range(1, epochs+1), cnn_bidirectional_lstm_accuracies, label='CNN-Bidirectional-LSTM', color = 'green')
    plt.plot(range(1, epochs+1), ann_accuracies, label='ANN', color = 'black')
    plt.title('Test Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend(loc='best')
    plt.show()


###GRAPHIC USER INTERFACE

#Extract features from the audio file
def process_audio_input(audio_file, resample_sr=8000, augment = False):
    audio = librosa.load(audio_file, sr=resample_sr)[0]
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=resample_sr, n_mfcc=40).T, axis=0)
    features = np.hstack((zcr, mfcc))
    return features.reshape(1, -1, 1)

def predict_emotion(audio_file, model_file='emotion_recognition_model.h5', augment=False):
    model = tf.keras.models.load_model(model_file)
    input_data = process_audio_input(audio_file, augment=augment)
    predictions = model.predict(input_data)
    emotions = ['Angry', 'Calm', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    emotion = emotions[np.argmax(predictions)]
    return emotion


# Function to record audio for a specified amount of seconds and save it to a .wav file.
def record_audio(seconds=3, filename='temp_audio.wav'):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Start recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * 10)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to record audio for a specified duration and save it to a temporary file.    
def record_and_save_temp_audio(seconds=3):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    record_audio(seconds=seconds, filename=temp_file.name)
    return temp_file.name

# Function to handle the button click event for recording audio.
def on_record_button_click():
    temp_audio_file = record_and_save_temp_audio()
    emotion = predict_emotion(temp_audio_file)

# Function to open an audio file dialog and return the selected file path.
def open_audio_file():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(None, "Open Audio File", "", "Audio Files (*.wav)", options=options)
    if file_name:
        return file_name
    return None    

# This class creates a new thread to record the audio so that the user interface remains responsive.
class RecordThread(QThread):
    update_timer = pyqtSignal(str)

    def run(self):
        self.temp_audio_file = record_and_save_temp_audio()
        emotion = predict_emotion(self.temp_audio_file)
        self.finished.emit()

    def start_timer(self, seconds):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.time_left = seconds
        self.timer.start(1000)

    def update_countdown(self):
        self.time_left -= 1
        self.update_timer.emit(f"{self.time_left:02d}")
        if self.time_left <= 0:
            self.timer.stop()

# Function to load an icon from a file and return a QIcon object.
def load_icon(file_name):
    return QIcon(file_name)

# This class defines the main application window.        
class AppDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.setStyleSheet("background-color: black; color: white;")

        self.setWindowTitle('Emotion Recognition System')

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()

        top_buttons_layout = QHBoxLayout()
        
        custom_font = QFont("Impact", 15)  

        self.record_button = QPushButton()
        self.record_button.setIcon(load_icon('/Users/aaryanjha/Desktop/Comp_Sci_Seminar/images/2.png'))
        self.record_button.setIconSize(self.record_button.size()*1) 
        self.record_button.clicked.connect(self.on_record_button_click)
        self.record_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        top_buttons_layout.addWidget(self.record_button)

        self.load_audio_button = QPushButton()
        self.load_audio_button.setIcon(load_icon('/Users/aaryanjha/Desktop/Comp_Sci_Seminar/images/1.png'))
        self.load_audio_button.setIconSize(self.load_audio_button.size()*1) 
        self.load_audio_button.clicked.connect(self.on_load_audio_button_click)
        self.load_audio_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        top_buttons_layout.addWidget(self.load_audio_button)

        main_layout.addLayout(top_buttons_layout)

        self.timer_label = QLabel("00:00")
        main_layout.addWidget(self.timer_label)
        self.timer_label.setFont(custom_font)


        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.emotion_label = QLabel('Detected emotion:')
        main_layout.addWidget(self.emotion_label)
        self.emotion_label.setFont(custom_font)


        self.recording_label = QLabel('Recording duration:')
        main_layout.addWidget(self.recording_label)
        self.recording_label.setFont(custom_font)

        bottom_layout = QHBoxLayout()

        bottom_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Fixed))

        self.exit_button = QPushButton()
        icon = load_icon('/Users/aaryanjha/Desktop/Comp_Sci_Seminar/images/3.png')
        self.exit_button.setIcon(icon)

        icon_size = QSize(55, 55)  

        self.exit_button.setIconSize(icon_size)

        self.exit_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.exit_button.clicked.connect(self.close)
        bottom_layout.addWidget(self.exit_button)

        main_layout.addLayout(bottom_layout)

        main_widget.setLayout(main_layout)

    def on_record_button_click(self):
        self.record_thread = RecordThread()
        self.record_thread.finished.connect(self.on_record_thread_finished)
        self.record_thread.update_timer.connect(self.update_timer_label)
        self.record_thread.start()
        self.record_thread.start_timer(5)

    def update_timer_label(self, time_left):
        self.timer_label.setText(f"{time_left}")
        
    def on_load_audio_button_click(self):
        audio_file = open_audio_file()
        if audio_file:
            emotion = predict_emotion(audio_file, augment = False)
            self.emotion_label.setText(f"Detected emotion: {emotion}")

    def on_record_thread_finished(self):
        if not sip.isdeleted(self.load_audio_button):
            self.load_audio_button.setEnabled(True)
        if not sip.isdeleted(self.record_button):
            self.record_button.setEnabled(True)
        temp_audio_file = self.record_thread.temp_audio_file
        emotion = predict_emotion(temp_audio_file)
        self.emotion_label.setText(f"Detected emotion: {emotion}")
        self.recording_label.setText(f"Recording duration: {5} seconds")

        
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_emotion_label(self, emotion):
        self.emotion_label.setText(f'Detected emotion: {emotion}')
        
    def closeEvent(self, event):
        if hasattr(self, 'record_thread'):
            self.record_thread.quit()
            self.record_thread.wait()
        event.accept()

def model_running():    
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot, encoder = preprocess()
    
    #display_emotion("neutral")
    #display_emotion("calm")
    #display_emotion("happy")
    #display_emotion("sad")
    #display_emotion("angry")
    #display_emotion("fear")
    #display_emotion("disgust")
    #display_emotion("surprise")
    
    svm_model, svm_test_accuracy = train_svm_model(X_train, X_test, y_train_onehot, y_test_onehot, encoder)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    cnn_lstm_model, cnn_lstm_test_accuracy, cnn_lstm_history= train_cnn_lstm_model(X_train, X_test, y_train_onehot, y_test_onehot)
    cnn_bidirectional_lstm_model, cnn_bidirectional_lstm_test_accuracy, cnn_bidirectional_lstm_history = train_cnn_bidirectional_lstm_model(X_train, X_test, y_train_onehot, y_test_onehot)
    ann_model, ann_test_accuracy, ann_history = train_ann_model(X_train, X_test, y_train_onehot, y_test_onehot)

    print("SVM Test Accuracy: ", svm_test_accuracy)
    print("CNN-LSTM Test Accuracy: ", cnn_lstm_test_accuracy)
    print("CNN BiDirectional LSTM Test Accuracy: ", cnn_bidirectional_lstm_test_accuracy)
    print("ANN Test Accuracy: ", ann_test_accuracy)

    
    plot_test_accuracies(cnn_lstm_history, svm_test_accuracy, cnn_bidirectional_lstm_history, ann_history)
    ann_model.save('emotion_recognition_model.h5')
    
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    model_running()    

