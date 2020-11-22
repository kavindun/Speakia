# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from deepface import DeepFace # import deepface library which is used to detect facial experssions
import cv2  #import opencv
import pyaudio
import wave
import noisereduce
from scipy.io import wavfile
import numpy as np
import multiprocessing
mysp=__import__("my-voice-analysis")

def facial_expersion(p3_emotion,lock,q):
    #speechRate=p1_rate.recv()
    speechRate=0

    # import haarcascade_frontalface_default.xml cascade file from Resources folder
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    faceDetect = cv2.CascadeClassifier('Resources/hand.xml')

    # open the web camera and specified camera id as id=0
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width set- 3=id and 640=value
    cap.set(4, 480)  # height set
    cap.set(10, 100)  # set brightness

    # web camer video is being read frame by frame In this while loop block
    # analise each frame.
    # detect faces and drawing rectangle using cascade file
    # analise face and detect facial experssion using deepface library functions
    while True:
        ret, frame = cap.read()
        # convert orginal frame to gray scale frame
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face and face scales using cascade file
        faces = faceCascade.detectMultiScale(frameGray, 1.1, 4)
        # get number of faces (lenth of faces object indicate number of faces)
        hands = faceDetect.detectMultiScale(frameGray, 1.1, 5);
        if len(faces) > 0:
            print("Hand Movement   : Detected")
            # p1_emotion.send(1)
        else:
            print("Hand Movement   : None")
            # p1_emotion.send(0)
        for (x, y, p, q) in hands:
            cv2.rectangle(frame, (x, y), (x + p, y + q), (0, 255, 0), 2)
        facesCount = len(faces)
        if facesCount >= 0:
            # draw  rectangle around facess
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            try:
                # facial experssion analyze using deepface library,for that purpose use orginal frame
                result = DeepFace.analyze(frame, actions=['emotion'])
                # print(result['dominant_emotion'])
                # write experison result as text in video frame
                emotion = "Facial Expression : " + result['dominant_emotion']
                print(emotion)
                cv2.putText(frame, emotion, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)
            except:
                emotion = "Facial Expression : None"
                print(emotion)
                cv2.putText(frame, emotion, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)
                pass
            cv2.imshow("output", frame)
            # this condition use to stop proccess when press Key Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def speech_rate(p2_rate):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44000
    CHUNK = 1024  # 512
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "recordedFile.wav"
    NOISE_AUDIO_CLIP = "noisepart1.wav"
    WAVE_INPUT_FILENAME = "noiselessrecord.wav"
    device_index = 2



    audio = pyaudio.PyAudio()

    def selectAudioInputDeviceIndex():
        print("----------------------record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

        print("-------------------------------------------------------------")
        print("Enter Device Index")
        index = int(input())
        print("recording via index " + str(index))
        return index

    def getAudioInput(index):
        audioInput = pyaudio.PyAudio()
        stream = audioInput.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=index,
                                 frames_per_buffer=CHUNK)
        # print("recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        # print("recording stopped")
        stream.stop_stream()
        stream.close()
        audioInput.terminate()
        return Recordframes

    def writeAudioFile(Recordframes):
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

    def reduceNoise(audio_file_path, noise_audio_file_path):
        rate, data = wavfile.read(audio_file_path)
        rate_noisy, noisy_part = wavfile.read(noise_audio_file_path)

        data = data.astype(np.float32)

        noisy_part = noisy_part.astype(np.float32)

        noise_reduce_data = noisereduce.reduce_noise(audio_clip=data, noise_clip=noisy_part, prop_decrease=1.0,
                                                     verbose=False)

        noise_reduce_data = noise_reduce_data.astype(np.int16)
        wavfile.write(filename=WAVE_OUTPUT_FILENAME, rate=rate, data=noise_reduce_data)

    #index = selectAudioInputDeviceIndex()
    print("select audio input and ID is 1")
    index=1
    while True:
        speechRate=-1
        p2_rate.send(speechRate)
        Recordframes = getAudioInput(index=index)
        writeAudioFile(Recordframes=Recordframes)
        reduceNoise(audio_file_path=WAVE_OUTPUT_FILENAME, noise_audio_file_path=NOISE_AUDIO_CLIP)

        p = "recordedFile"  # Audio File title
        c = r"E:\SJP\3rd Year\2ndSemestar\Cs\HCI\Public Speech App\Refferancess\public_speech_app"  # Path to the Audio_File directory
        speechRate=mysp.myspsr(p, c)
        print("Speech Rate     : ",speechRate)
        #p2_rate.send(speechRate)


        # print(ss)

def hand_gesture_detection(p1_rate,p1_emotion,lock,q):
    faceDetect = cv2.CascadeClassifier('Resources/hand.xml')
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width set- 3=id and 640=value
    cap.set(4, 480)  # height set
    cap.set(10, 100)  # set brightness


    while (True):
        #lock.acquire()
        #cap = cv2.VideoCapture(0);
        ret, frame = cap.read();
        #p1_emotion.send(frame)
        q.put(frame)
        #cap.release()
        #lock.release()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = faceDetect.detectMultiScale(gray, 1.1, 5);
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        if len(faces)>0:
            print("Hand Movement   : Detected")
            #p1_emotion.send(1)
        else:
            print("Hand Movement   : None")
            #p1_emotion.send(0)
        for (x, y, p, q) in hands:
            cv2.rectangle(frame, (x, y), (x + p, y + q), (0, 255, 0), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        #speechRate=p1_rate.recv()

        #strinSpeeRate = "speech rate = " + str(speechRate)
        #cv2.putText(frame, strinSpeeRate, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)
        cv2.imshow("Face", frame)
        #lock.release()
        if (cv2.waitKey(1) == ord('q')):
            break;
    cap.release()
    cv2.destroyAllWindows()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #facial_expersion()
    #speech_rate()
    #cap = cv2.VideoCapture(0)
    p1_rate,p2_rate=multiprocessing.Pipe()
    p1_emotion,p3_emotion=multiprocessing.Pipe()
    q=multiprocessing.Queue()
    lock=multiprocessing.Lock()

    p3=multiprocessing.Process(target=facial_expersion,args=(p3_emotion,lock,q,))
    p2=multiprocessing.Process(target=speech_rate,args=(p2_rate,))
    p1=multiprocessing.Process(target=hand_gesture_detection,args=(p1_rate,p1_emotion,lock,q,))
    print("proceess p1 start")
    #p1.start()
    print("process p2 satrt")
    p2.start()
    print("proccess p3 start")
    p3.start()
    print("runing...")
    #p1.join()
    p2.join()
    p3.join()
    #cap.release()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
