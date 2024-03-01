import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pickle
import streamlit as st
import speech_recognition as sr
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import string
import time


#TITLE USING STREAMLIT
st.title("SIGN LANGUAGE TRANSLATOR")


selected_option = st.selectbox('Choose',('sign to text','audio to sign'))


if selected_option == 'sign to text':

    clf = joblib.load('model.pkl')
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    #confidence level drops below 50% then the hands will not be detected at all in the output image.
    #tacking confidence is the threshold value between current frame and last frame if it is less than 50% it doesmnt detect
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


    def data_clean(landmark):
        # Clean the landmark data

        # Extract data from the input landmark list
        data = landmark[0]
        try:
            # Convert data to string
            data = str(data)
            # Split data by newline character and remove leading/trailing whitespaces
            data = data.strip().split('\n')
            # Define a list of garbage strings to be removed from data
            garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
            without_garbage = []
            # Remove garbage strings from data
            for i in data:
                if i not in garbage:
                    without_garbage.append(i)
            clean = []
            # Remove leading whitespaces from each string in without_garbage and append to clean
            for i in without_garbage:
                i = i.strip()
                # Append substring starting from the third character to clean
                clean.append(i[2:])
            finalClean = []
            # Iterate over clean list and convert each element to float, skipping every third element
            for i in range(0, len(clean)):
                if (i + 1) % 3 != 0:
                    finalClean.append(float(clean[i]))
            # Return the final cleaned data as a list of lists
            return ([finalClean])
        except:
            # If an exception occurs, return a numpy array of zeros with shape (1, 63)
            return (np.zeros([1, 63], dtype=int)[0])


    def process(image):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Setting image.flags.writeable = False is used to prevent the modification of image data.In this  it is used before passing the
        #image to the hands.process() function, which likely uses the image data for processing but does not need to modify it.
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_Lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_Lms,
                    mp_hands.HAND_CONNECTIONS)
                cleaned_landmark = data_clean(results.multi_hand_landmarks)
                y_pred = clf.predict(cleaned_landmark)
                image = cv2.putText(image, str(y_pred[0]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        return image

#To  establish a peer-to-peer connection - webrtc is used to connect users in real time .
# WebRTC connections can't run without a server
#so a stun server is used to connect in real time
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


class VideoProcessor:
    def recv(self, frame):
        #image frame obtained is converted to numpy array
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

 # webrtc streaner is Streamlit component which deals with video and audio real-time I/O through web browsers.
#The key argument is a unique ID in the script to identify the component instance.
webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,

    )

if selected_option == 'audio to sign':
    start_button = st.button("Start Microphone Input")
    if start_button:
        def func():
            r = sr.Recognizer()
            arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            # source = microphone by which we give input
            with sr.Microphone(device_index=1) as source:

                # r.adjust_for_ambient_noise(source)
                i = 0
                while True:
                    st.write('Say something')
                    # to remove the backgroundnoise
                    r.adjust_for_ambient_noise(source)
                    # audio= the audio signal which we speak
                    audio = r.listen(source)
                    try:
                        # speech to text conversion
                        a = r.recognize_google(audio)
                        st.write("you said " + format(a))
                        for c in string.punctuation:
                            a = a.replace(c, "")

                        if (a.lower() == 'goodbye' or a.lower() == 'good bye' or a.lower() == 'bye'):
                            st.write("oops!Time To say good bye")
                            break

                        else:
                            for i in range(len(a)):
                                if a[i] in arr:


                                    ImageAddress = 'letters/' + a[i] + '.jpg'
                                    ImageItself = Image.open(ImageAddress)
                                    # convert the image in numpy array format to plot
                                    ImageNumpyFormat = np.asarray(ImageItself)
                                    st.image(ImageNumpyFormat, width=200, caption=a[i])
                                    time.sleep(0.05)
                                else:
                                    continue

                    except:
                        st.write(" ")
                    plt.close()


        while 1:
            image = "signlang.jpeg"
            msg = "HEARING IMPAIRMENT ASSISTANT"

            # if live audio is selected then go to func else e
            func()