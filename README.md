# Steps To Run .sav as well as .h5 Models - <br>
1. Clone the repository- <br>
```
git clone https://github.ecodesamsung.com/SRIB-PRISM/SRM_23VI26_Age_gender_and_emotion_detection_from_speech.git
```
2. Navigate to the project directory- <br>
```
cd SRM_23VI26_Age_gender_and_emotion_detection_from_speech
```
3. Install dependencies-
```
pip install -r requirements.txt
```
4. Download the '.h5' or '.sav' model from the repository.
5. Run the inference script-
```
python inference_script.py --audio_path sample_audio/file.wav --model_path model_name.h5/model_name.sav
``` 
<br>
