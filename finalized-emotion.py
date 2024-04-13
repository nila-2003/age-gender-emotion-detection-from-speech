#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # 1. Gathering data
# 
# We will gather speech data from four datasets and store it in a single dataframe along with the corresponding file paths, gender labels and emotion labels.

# In[ ]:


import os
import pandas as pd
import numpy as np


# In[ ]:


RAVDESS = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
CREMA = "/kaggle/input/cremad/AudioWAV/"
TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
SAVEE = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"


# **1. Ravdess Dataframe**
# 
# There are 1440 audio files, for example, 03-01-**06**-01-02-01-12.wav.

# In[ ]:


ravdess_dir_lis = os.listdir(RAVDESS)
path_list = []
gender_list = []
emotion_list = []

emotion_dic = {
    '03' : 'happy',
    '01' : 'neutral',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fear',
    '07' : 'disgust',
}

for directory in ravdess_dir_lis:
    actor_files = os.listdir(os.path.join(RAVDESS, directory))
    for audio_file in actor_files: 
        part = audio_file.split('.')[0]
        key = part.split('-')[2]
        if key in emotion_dic:
            gender_code = int(part.split('-')[6])
            path_list.append(f"{RAVDESS}{directory}/{audio_file}")
            gender_list.append('female' if gender_code & 1 == 0 else 'male')
            emotion_list.append(emotion_dic[key])
            
ravdess_df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(gender_list, columns=['sex']),
    pd.DataFrame(emotion_list, columns=['emotion'])
], axis=1)

ravdess_df.head()


# **2. Crema-D Dataframe**
# 
# There are 7,442 audio files, for example, 1001_DFA_**ANG**_XX.wav.

# In[ ]:


crema_dir_list = os.listdir(CREMA)
path_list = []
gender_list = []
emotion_list = []

emotion_dic = {
    'HAP' : 'happy',
    'NEU' : 'neutral',
    'SAD' : 'sad',
    'ANG' : 'angry',
    'FEA' : 'fear',
    'DIS' : 'disgust',
}

female_id_list = [
    '1002', '1003', '1004', '1006', '1007', '1008', '1009', '1010', '1012', '1013', '1018', 
    '1020', '1021', '1024', '1025', '1028', '1029', '1030', '1037', '1043', '1046', '1047', 
    '1049', '1052', '1053', '1054', '1055', '1056', '1058', '1060', '1061', '1063', '1072', 
    '1073', '1074', '1075', '1076', '1078', '1079', '1082', '1084', '1089', '1091',
]

for audio_file in crema_dir_list:
    part = audio_file.split('_')
    key = part[2]
    if key in emotion_dic and part[3] == 'HI.wav':
        path_list.append(f"{CREMA}{audio_file}")
        gender_list.append('female' if part[0] in female_id_list else 'male')
        emotion_list.append(emotion_dic[key])

crema_df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(gender_list, columns=['sex']),
    pd.DataFrame(emotion_list, columns=['emotion'])
], axis=1)

crema_df.head()


# **3. Tess Dataframe**
# 
# There are 2,800 audio files, for example, OAF_base_**fear**.wav.

# In[ ]:


tess_dir_list = os.listdir(TESS)
path_list = []
gender_list = []
emotion_list = [] 

emotion_dic = {
    'happy'   : 'happy',
    'neutral' : 'neutral',
    'sad'     : 'sad',
    'Sad'     : 'sad',
    'angry'   : 'angry',
    'fear'    : 'fear',
    'disgust'  : 'disgust',
}

for directory in tess_dir_list:
    audio_files = os.listdir(os.path.join(TESS, directory))
    for audio_file in audio_files:
        part = audio_file.split('.')[0]
        key = part.split('_')[2]
        if key in emotion_dic:
            path_list.append(f"{TESS}{directory}/{audio_file}") 
            gender_list.append('female') # female only dataset
            emotion_list.append(emotion_dic[key])
            
tess_df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(gender_list, columns=['sex']),
    pd.DataFrame(emotion_list, columns=['emotion'])
], axis=1)

tess_df.head()


# **4. Savee Dataframe**
# 
# There are 480 audio files, for example, DC_**a**02.wav.

# In[ ]:


savee_dir_list = os.listdir(SAVEE)
path_list = []
gender_list = []
emotion_list = []

emotion_dic = {
    'h'  : 'happy',
    'n'  : 'neutral',
    'sa' : 'sad',
    'a'  : 'angry',
    'f'  : 'fear',
    'd'  : 'disgust'
}

for audio_file in savee_dir_list:
    part = audio_file.split('_')[1]
    key = part[:-6]
    if key in emotion_dic:
        path_list.append(f"{SAVEE}{audio_file}")
        gender_list.append('male') # male only dataset
        emotion_list.append(emotion_dic[key])
        
savee_df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(gender_list, columns=['sex']),
    pd.DataFrame(emotion_list, columns=['emotion'])
], axis=1)

savee_df.head()


# In[ ]:


savee_df.shape


# In[ ]:


df = pd.concat([
    ravdess_df, 
    crema_df, 
    tess_df, 
    savee_df
], axis=0)
df.head()


# # 2. Quick EDA
# 
# We check for imbalances like male to female ratio. 

# In[ ]:


import librosa
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt

def plot_distribution(df): 
    countTable = df.groupby(['emotion', 'sex']).count()
    pivotTable = countTable.pivot_table(index='emotion', columns='sex', values='path')

    pivotTable.plot(kind='bar', figsize=(10, 5), color=['orange', 'blue'])
    plt.title('Emotion and Gender Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')

    # Save the plot as a PNG file
    plt.savefig('/kaggle/working/emotion_gender_distribution.png')
    
    # Display the plot
    plt.show()

# Assuming df is your DataFrame
plot_distribution(df)


# In[ ]:


df = df[df['sex'] == 'female']
plot_distribution(df)


# In[ ]:


df.drop('sex', axis=1, inplace=True)
df.head()


# In[ ]:


df['emotion'].value_counts()


# In[ ]:


from IPython.display import Audio

def create_waveplot(y, sr, title, filename):
    plt.figure(figsize=(8, 2))
    plt.title(title)
    librosa.display.waveshow(y, sr=sr)
    plt.savefig(filename)
    plt.show()


# In[ ]:


emotion_label = 'sad'
title = f"Waveplot for {emotion_label} emotion"
path = np.array(df.path[df.emotion == emotion_label])[1]
y, sr = librosa.load(path)

# Specify the filename to save the plot
filename = '/kaggle/working/waveplot.png'

create_waveplot(y, sr, title, filename)

# Display the audio
Audio(path)


# In[ ]:


emotion_label = 'happy'
title = f"Waveplot for {emotion_label} emotion"
path = np.array(df.path[df.emotion == emotion_label])[1]
y, sr = librosa.load(path)

# Specify the filename to save the plot
filename = f'/kaggle/working/{emotion_label}_wave.png'

create_waveplot(y, sr, title, filename)

# Display the audio
Audio(path)


# In[ ]:


emotion_label = 'happy'
title = f"Waveplot for {emotion_label} emotion"
path = np.array(df.path[df.emotion == emotion_label])[1]
y, sr = librosa.load(path)

# Specify the filename to save the plot
filename = f'/kaggle/working/{emotion_label}_wave.png'

create_waveplot(y, sr, title, filename)

# Display the audio
Audio(path)


# # 3. Preprocess
# 
# The following steps will be followed to preprocess the audio: 
# 
# 1. Get an array of samples
# 2. Trim the silence  
# 3. Padding for equal length

# In[ ]:


from pydub import AudioSegment, effects


# In[ ]:


def preprocess_audio(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
    return padded, sr


# In[ ]:


emotion_dic = {
    'neutral' : 0,
    'happy'   : 1,
    'sad'     : 2, 
    'angry'   : 3, 
    'fear'    : 4, 
    'disgust' : 5
}

def encode(label):
    return emotion_dic.get(label)


# # 4. Extract features
# 
# We will only extract these features: 
# 
# 1. Mel-Frequency Cepstral Coefficients: captures the shape of the spectral envelope of a signal
# 2. Zero Crossing Rate: captures the number of times a signal changes sign per second
# 3. Root Mean Square Energy: captures the root mean square amplitude of the audio signal

# In[ ]:


zcr_list = []
rms_list = []
mfccs_list = []
emotion_list = []

FRAME_LENGTH = 2048
HOP_LENGTH = 512

for row in df.itertuples(index=False):
    try: 
        y, sr = preprocess_audio(row.path)

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

        zcr_list.append(zcr)
        rms_list.append(rms)
        mfccs_list.append(mfccs)

        emotion_list.append(encode(row.emotion))
    except:
        print(f"Failed for path: {row.path}")


# In[ ]:


X = np.concatenate((
    np.swapaxes(zcr_list, 1, 2), 
    np.swapaxes(rms_list, 1, 2), 
    np.swapaxes(mfccs_list, 1, 2)), 
    axis=2
)
X = X.astype('float32')

y = np.asarray(emotion_list)
y = np.expand_dims(y, axis=1).astype('int8')


# # 5. Build a LSTM
# 
# Before building the model, we will have to setup the data. LSTM are great for sequences.  

# In[ ]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[ ]:


X_train, X_to_split, y_train, y_to_split = train_test_split(X, y, test_size=0.12, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_to_split, y_to_split, test_size=0.3, random_state=1)

y_train_class = to_categorical(y_train, 6)
y_val_class = to_categorical(y_val, 6)


# In[ ]:


from keras.models import Sequential
from keras import layers, optimizers, callbacks


# In[ ]:


MODEL = Sequential()
MODEL.add(layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1:3])))
MODEL.add(layers.LSTM(64))
MODEL.add(layers.Dense(6, activation='softmax'))

print(MODEL.summary())


# In[ ]:


rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.01, patience=100)
MODEL.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])
model = MODEL.fit(X_train, y_train_class, epochs=200, batch_size=6, validation_data=(X_val, y_val_class))


# # 6. Evaluate and conclude
# 
# Let's see how good are model is. 

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(model.history['loss'])
axes[0].plot(model.history['val_loss'])
axes[0].set_title('Loss for Train and Validation Sets')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epochs')
axes[0].legend(['Training', 'Validation'])

axes[1].plot(model.history['categorical_accuracy'])
axes[1].plot(model.history['val_categorical_accuracy'])
axes[1].set_title('Accuracy for Train and Validation Sets')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epochs')
axes[1].legend(['Training', 'Validation'])

fig.tight_layout()

plt.show()


# In[ ]:


import seaborn as sns

y_pred = np.argmax(MODEL.predict(X_val), axis=1)
labels = ['neutral', 'calm', 'sad', 'happy', 'fear', 'disgust']
cm = confusion_matrix(np.argmax(y_val_class, axis=1), y_pred, labels=range(6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()


# In[ ]:


MODEL1 = Sequential()
MODEL1.add(layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1:3])))
MODEL1.add(layers.LSTM(64))
MODEL1.add(layers.Dense(128, activation='relu'))
MODEL1.add(layers.Dense(6, activation='softmax'))

print(MODEL1.summary())


# In[ ]:


rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.01, patience=100)
MODEL1.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])
model = MODEL1.fit(X_train, y_train_class, epochs=100, batch_size=6, validation_data=(X_val, y_val_class))


# In[ ]:


MODEL2 = Sequential()
MODEL2.add(layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1:3])))
MODEL2.add(layers.LSTM(128))
MODEL2.add(layers.Dense(128, activation='relu'))
MODEL2.add(layers.Dense(64, activation='relu'))
MODEL2.add(layers.Dense(6, activation='softmax'))

print(MODEL2.summary())


# In[ ]:


rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.01, patience=100)
MODEL2.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])
model = MODEL2.fit(X_train, y_train_class, epochs=100, batch_size=6, validation_data=(X_val, y_val_class))


# **Conclusion**
# 
# In summary, our model achieved 90% accuracy in classifying six different emotions.

# In[ ]:




