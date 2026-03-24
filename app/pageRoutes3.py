from flask import render_template, Blueprint, Response, request
import cv2, time, base64
from app.extensions import socketio  # import the shared socketio instance

import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from threading import Lock
from app.utils import mediapipeDetection, extractFaceLandmarks, extractPoseLandmarks, checkVelocity, addVelocity, faceNormalization

import os
import time


bp = Blueprint("pageNames", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_PATH = os.path.join(BASE_DIR, "modelAction2.keras")
FACE_PATH = os.path.join(BASE_DIR, "moodFold2.keras")

actionModel = tf.keras.models.load_model(ACTION_PATH)
faceModel = tf.keras.models.load_model(FACE_PATH)

actionLabels = np.array([
  "goodMorning","goodAfternoon","goodEvening","hello",
  "howAreYou","imFine","niceToMeetYou","thankYou",
  "youreWelcome","seeYouTomorrow"
])
faceLabels = np.array(["neutral", "question"])

# warm up models
_ = actionModel(tf.zeros((1, 40, 516)), training=False)
_ = faceModel(tf.zeros((1, 1412,)), training=False)

ACTION_FRAME_LIMIT = 40
CONFIDENCE_LEVEL = 0.75

sessions = {}  # all per-user state lives here


def get_session(sid):
    if sid not in sessions:
        sessions[sid] = {
            "holistic": mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
                smooth_landmarks=True
            ),
            "actionContainer": deque(maxlen=ACTION_FRAME_LIMIT),
            "prevLmk": None,
            "frameCounter": 0,
            "facePrediction": None,
            "faceConfidence": None,
            "faceVotes": [],        # collect 5 face predictions for majority vote
            "lastSentWord": None,
            "lastInferenceFrame": 0  # track when last inference happened
        }
    return sessions[sid]

@socketio.on("connect")
def handle_connect():
    get_session(request.sid)
    print(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in sessions:
        sessions[sid]["holistic"].close()  # release mediapipe instance
        del sessions[sid]
    print(f"Client disconnected: {sid}")

@socketio.on("clear_session")
def handle_clear():
    sid = request.sid
    if sid in sessions:
        sessions[sid]["actionContainer"].clear()
        sessions[sid]["frameCounter"] = 0
        sessions[sid]["lastSentWord"] = None


@socketio.on("video_frame")
def handle_video_frames(data):
    sid = request.sid
    session = get_session(sid)

    frame_b64 = data["frame"].split(",")[1]
    frame_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is not None:

        _, results = mediapipeDetection(frame, session["holistic"])
        del frame  # discard immediately

        if session["frameCounter"] % 2 == 0:
            actionLmk = extractPoseLandmarks(results)

            if session["prevLmk"] is not None and checkVelocity(actionLmk, session["prevLmk"]):
                session["actionContainer"].append(actionLmk)
                session["isGesturing"] = True
            session["prevLmk"] = actionLmk

        if session["frameCounter"] % 8 == 0:
            faceResults = extractFaceLandmarks(results)
            faceLmk = faceNormalization(faceResults)

            input_tensor = tf.convert_to_tensor(np.expand_dims(faceLmk, axis=0), dtype=tf.float32)
            facePrediction = faceModel(input_tensor, training=False)
            facePrediction = facePrediction.numpy()[0][0]

            session["faceVotes"].append(facePrediction)

            if len(session["faceVotes"]) > 5:
                session["faceVotes"].pop(0)

        session["frameCounter"] += 1

        if (len(session["actionContainer"]) == ACTION_FRAME_LIMIT):
            snapshot = addVelocity(np.array(session["actionContainer"]))  # fixed
            input_tensor = tf.convert_to_tensor(np.expand_dims(snapshot, axis=0), dtype=tf.float32)
            actionPrediction = actionModel(input_tensor, training=False)
            actionPrediction = actionPrediction.numpy()
            actionConfidence = actionPrediction.max()

            if actionConfidence >= CONFIDENCE_LEVEL:
                actionWord = actionLabels[np.argmax(actionPrediction)]

                if session["lastSentWord"] != actionWord:
                    session["lastSentWord"] = actionWord

                    # reset face votes after sending
                    session["actionContainer"].clear()  # fresh start after prediction
                    session["prevLmk"] = None

                    socketio.emit("translation", {
                        "word": actionWord,
                        "confidence": float(actionConfidence)
                    }, room=sid)

            if len(session["faceVotes"]) >= 5:
                # majority vote on face
                if session["faceVotes"]:
                    avg_face = sum(session["faceVotes"]) / len(session["faceVotes"])
                    mood_result = faceLabels[int(avg_face > 0.95)]
                    mood_confidence = abs(avg_face - 0.5) * 2
                else:
                    mood_result = "neutral"
                    mood_confidence = 0.0

                # question_votes = sum(1 for v in session["faceVotes"] if v > 0.85)
                # mood_result = faceLabels[1] if question_votes >= 4 else faceLabels[0]

                for v in session["faceVotes"]: print(v)

                session["faceVotes"] = []
                socketio.emit("face", {
                    "mood": mood_result,
                    "confidence": float(mood_confidence)
                }, room=sid)

    socketio.emit("ack", room=sid)  # tell client we're done

@bp.route('/')
def home():
    return render_template("pageNames/home.html")

@bp.route('/about')
def about():
    return render_template("pageNames/about.html")

@bp.route('/camera')
def camera():
    return render_template("pageNames/camera3.html")
