# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from deepface import DeepFace # import deepface library which is used to detect facial experssions
import cv2 #import opencv
import pyaudio
import wave
import noisereduce
from scipy.io import wavfile
import numpy as np
import multiprocessing
import argparse
import threading
import time
import speech_recognition as sr
from os import path
from textblob import TextBlob
from gaze_tracking import GazeTracking
mysp=__import__("my-voice-analysis")


timerFlag=False
timeHandFold=0
timerLoopFlag=True
gFrame=[]
textOfSpeech=""

facialExpersionsPresentage=[0,0,0,0,0,0,0] #this array have presentage values of facial experssions.index order of experssions :[happy,sad,netrul,angry,disgust,suprice,fear]

speechSentimentalAnalysis="none" #this variable have string value which denote that sentimental analysis of speech,the values may be "none" or "positive" or "negative" or "neutral"

eyeContactAreaPresentage=[0,0,0,0,0,0] #presentage value array of eye direction area
"""
Index   Related Area
0       Left Far
1       Left Near
2       Right Far
3       Right Near
4       Centera Far
5       Centar Near
"""
handGesturePositionPresentage=[0,0,0,0] #hand gesture positon as presentages value
"""
index   gesture
0       hand above shoulder(positive)
1       hand near to shoulder(neutral)
2       open hand belove shoulde (positive)
3       close hand belove shoulder (negative)

"""
ahCountArray=[5,3,0,2,7,4,9,5]
def eyeGazeTracking(frame,eyeContactAreaCounts,totalEyeContactCounts):

    eyeContactAreaCounts=[10,100,5,150,110,250]
    totalEyeContactCounts=10+100+5+150+110+250
    # We send this frame to GazeTracking to analyze it
    print("gaze")
    #gaze=GazeTracking()
    print("frame")

    #gaze.refresh(frame)
    print("anota")
    #frame = gaze.annotated_frame()
    print("condition")
    """
    if gaze.is_left():
        totalEyeContactCounts = totalEyeContactCounts + 1
        if gaze.vertical_ratio()>=0.5:
            eyeContactAreaCounts[1] = eyeContactAreaCounts[1] + 1
            print("left Near")
        elif gaze.vertical_ratio()<0.5:
            eyeContactAreaCounts[0]=eyeContactAreaCounts[0]+1
            print("left far")
        else:
            pass
    elif gaze.is_right():
        totalEyeContactCounts = totalEyeContactCounts + 1
        if gaze.vertical_ratio()>=0.5:
            eyeContactAreaCounts[3] = eyeContactAreaCounts[3] + 1
            print("right Near")
        elif gaze.vertical_ratio()<0.5:
            eyeContactAreaCounts[2] = eyeContactAreaCounts[2] + 1
            print("right Far")
        else:
            pass
    elif gaze.is_center():
        totalEyeContactCounts = totalEyeContactCounts + 1
        if gaze.vertical_ratio()>=0.5:
            eyeContactAreaCounts[5] = eyeContactAreaCounts[5] + 1
            print("center Near")
        elif gaze.vertical_ratio()<0.5:
            eyeContactAreaCounts[4] = eyeContactAreaCounts[4] + 1
            print("center Far")
        else:
            pass
    else:
        pass
        """


    return eyeContactAreaCounts,totalEyeContactCounts


def rightHandFold(points):
    if points[2] and points[4]:
        if points[4][0] > points[2][0]:
            print("right hand fold ")
            return True
        else:
            return False
    else:
        return False

def leftHandFold(points):
    print("wrist",points[7])
    print("sholder",points[5])
    if points[7] and points[5]:
        if points[7][0] < points[5][0]:
            print("left hand fold ")
            return True
        else:
            return False
    else:
        return False

def isHandFold(points):
    if rightHandFold(points) and leftHandFold(points):
        return True
    else:
        return False

def handGesture(points, gestureCounts, totalGestureCounts):
    if points[2] and points[5] and points[4] and points[7]:


        averageShoulderX = (points[2][1] + points[5][1]) / 2
        if (averageShoulderX == points[4][1]) and averageShoulderX == points[7][1]:
            totalGestureCounts=totalGestureCounts+1
            gestureCounts[0]=gestureCounts[0]+1
            print("near shoulder")
        elif (averageShoulderX > points[4][1]) and averageShoulderX > points[7][1]:
            totalGestureCounts = totalGestureCounts + 1
            print("shoulder below")
            if isHandFold(points):
                gestureCounts[3]=gestureCounts[3]+1
                print("hand fold")
            else:
                gestureCounts[2]=gestureCounts[2]+1
                print("hand open")
        elif (averageShoulderX < points[4][1]) and averageShoulderX < points[7][1]:
            totalGestureCounts = totalGestureCounts + 1
            gestureCounts[1]=gestureCounts[1]+1
            print("hand above shoulder")
        else:
            pass
    else:
        pass
    return gestureCounts,totalGestureCounts



def poseEstimate():
    global timerFlag
    global gFrame
    global eyeContactAreaPresentage
    global handGesturePositionPresentage

    eyeContactAreaCounts=[0,0,0,0,0,0]
    totalEyeContactCounts = 0
    gestureCounts=[0,0,0,0]
    totalGestureCounts=0

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

    args = parser.parse_args()

    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    inWidth = args.width
    inHeight = args.height

    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

    cap = cv2.VideoCapture(args.input if args.input else 0)

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        gFrame=frame
        eyeContactAreaCounts, totalEyeContactCounts = eyeGazeTracking(frame, eyeContactAreaCounts,
                                                                      totalEyeContactCounts)
        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(
            cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)
        if isHandFold(points):
            print("hand fold occure")
            timerFlag=True
        else:
            timerFlag=False
        gestureCounts,totalGestureCounts=handGesture(points,gestureCounts,totalGestureCounts)
        #eyeContactAreaCounts, totalEyeContactCounts = eyeGazeTracking(frame, eyeContactAreaCounts,totalEyeContactCounts)

        #leftHandFold(points)
        #rightHandFold(points)
        #print(points[0])
        #print(points[1])

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        print("get")
        #eyeContactAreaCounts,totalEyeContactCounts=eyeGazeTracking(frame,eyeContactAreaCounts,totalEyeContactCounts)
        cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('OpenPose using OpenCV', frame)
    for i in range(0,len(eyeContactAreaPresentage)):
        if totalEyeContactCounts>0:
            eyeContactAreaPresentage[i]=eyeContactAreaCounts[i]/totalEyeContactCounts*100
    for i in range(0, len(handGesturePositionPresentage)):
        if totalGestureCounts>0:
            handGesturePositionPresentage[i] = gestureCounts[i] / totalGestureCounts * 100


def facial_expersion(): #p3_emotion,lock,q
    #speechRate=p1_rate.recv()
    speechRate=0
    global gFrame
    global timerLoopFlag
    global facialExpersionsPresentage

    frameCountExpressions=[0,0,0,0,0,0,0]
    frameCount=0

    # import haarcascade_frontalface_default.xml cascade file from Resources folder
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    faceDetect = cv2.CascadeClassifier('Resources/hand.xml')

    # open the web camera and specified camera id as id=0
    # web camer video is being read frame by frame In this while loop block
    # analise each frame.
    # detect faces and drawing rectangle using cascade file
    # analise face and detect facial experssion using deepface library functions
    while timerLoopFlag:
        try:
            # facial experssion analyze using deepface library,for that purpose use orginal frame
            result = DeepFace.analyze(gFrame, actions=['emotion'])
            frameCount=frameCount+1
            if result['dominant_emotion']=="happy":
                frameCountExpressions[0]=frameCountExpressions[0]+1
            elif result['dominant_emotion']=="sad":
                frameCountExpressions[1]=frameCountExpressions[1]+1
            elif result['dominant_emotion']=="neutral":
                frameCountExpressions[2]=frameCountExpressions[2]+1
            elif result['dominant_emotion']=="angry":
                frameCountExpressions[3] = frameCountExpressions[3] + 1
            elif result['dominant_emotion'] == "disgust":
                frameCountExpressions[4] = frameCountExpressions[4] + 1
            elif result['dominant_emotion'] == "surprise":
                frameCountExpressions[5] = frameCountExpressions[5] + 1
            elif result['dominant_emotion'] == "fear":
                frameCountExpressions[6] = frameCountExpressions[6] + 1



            # print(result['dominant_emotion'])
            # write experison result as text in video frame
            emotion = "Facial Expression : " + result['dominant_emotion']
            print(emotion)
            #cv2.putText(frame, emotion, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)
        except:
            emotion = "Facial Expression : None"
            print(emotion)
            #cv2.putText(frame, emotion, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)
            pass
        #cv2.imshow("output", frame)
        # this condition use to stop proccess when press Key Q
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    print("facial expression loop end")
    for i in range(0,len(facialExpersionsPresentage)):
        facialExpersionsPresentage[i]=frameCountExpressions[i]/frameCount*100
        print("fep : ",frameCountExpressions[i])
    print("fc :",frameCount)

def ahCount():
    return

def speech_rate():#p2_rate
    global textOfSpeech
    global timerLoopFlag
    global speechSentimentalAnalysis


    tFlag=timerLoopFlag
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44000
    CHUNK = 1024  # 512
    RECORD_SECONDS = 3600
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
        global timerLoopFlag
        audioInput = pyaudio.PyAudio()
        stream = audioInput.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=index,frames_per_buffer=CHUNK)
        Recordframes = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            print("audio loop going")
            if not timerLoopFlag:
                break
            data = stream.read(CHUNK)
            Recordframes.append(data)
        # print("recording stopped")

        """
        while timerLoopFlag:
            print("1")
            try:
                data = stream.read(CHUNK)
            except:
                pass
            print("2")
            Recordframes.append(data)
            print("3")
        """
        print("auido loop end")
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

        noise_reduce_data = noisereduce.reduce_noise(audio_clip=data, noise_clip=noisy_part, prop_decrease=1.0, verbose=False)

        noise_reduce_data = noise_reduce_data.astype(np.int16)
        wavfile.write(filename=WAVE_OUTPUT_FILENAME, rate=rate, data=noise_reduce_data)

    #index = selectAudioInputDeviceIndex()
    print("select audio input and ID is 1")
    index=1

    Recordframes = getAudioInput(index=index)
    writeAudioFile(Recordframes=Recordframes)
    reduceNoise(audio_file_path=WAVE_OUTPUT_FILENAME, noise_audio_file_path=NOISE_AUDIO_CLIP)
    text=speech_to_text("recordedFile.wav")
    print("text :",text)
    if not text=="":
        speechSentimentalAnalysis=sentimental_analysise(text)
    ahCount()



def sentimental_analysise(text):
    analysise = TextBlob(text)
    polarityScore=analysise.sentiment.polarity
    if polarityScore>0.3:
        return "positive"
    elif polarityScore<-0.3:
        return "negative"
    else:
        return "neutral"


def speech_to_text(recordedFilePath):
    r1 = sr.Recognizer()
    with sr.AudioFile(recordedFilePath) as source:  # OSR_us_000_0010_8k.wav
        print("listing")
        audioSpeech = r1.record(source)

    try:
        text = r1.recognize_google(audioSpeech)
        print(text)
        return text
        analysise = TextBlob(text)
        print(analysise.sentiment.polarity)
    except:
        print("none speech")
        text=""
        return text;
        pass



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')
    print(len(gFrame))


    t2 = threading.Thread(target=poseEstimate)
    t3=threading.Thread(target=facial_expersion)
    t4=threading.Thread(target=speech_rate)

    t2.start()
    t3.start()
    time.sleep(10)
    t4.start()
    t2.join()
    timerLoopFlag=False
    t3.join()
    
    t4.join()
    print("text of speech :",textOfSpeech)
    print("sentimetal analysis :",speechSentimentalAnalysis)
    print("speech presentage:")

    for i in facialExpersionsPresentage:
        print(i)
    print("eye contact")
    for p in eyeContactAreaPresentage:
        print(p)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
