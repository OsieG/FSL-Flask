import cv2
import mediapipe as mp
import numpy as np

mpHolistic = mp.solutions.holistic     #Holistic Model
mpDrawing = mp.solutions.drawing_utils #Drawing Utilities

def mediapipeDetection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def drawStyledLandmarks(image, results):
    mpDrawing.draw_landmarks(
        image, results.face_landmarks, mpHolistic.FACEMESH_TESSELATION,
        mpDrawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    mpDrawing.draw_landmarks(
        image, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
        mpDrawing.DrawingSpec(color=(210,166,205), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(150,156,170), thickness=1, circle_radius=1)
    )
    mpDrawing.draw_landmarks(
        image, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
        mpDrawing.DrawingSpec(color=(142,202,230), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(33,158,188), thickness=1, circle_radius=1)
    )
    mpDrawing.draw_landmarks(
        image, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
        mpDrawing.DrawingSpec(color=(148,190,71), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(91,168,160), thickness=1, circle_radius=1)
    )

def extractLandmarks(results):
    faceLmk = np.zeros((468,3), dtype=np.float32)
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            faceLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    poseLmk = np.zeros((33,4), dtype=np.float32)
    if results.pose_landmarks:
        offset = 0
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            poseLmk[i] = np.array([landmark.x, landmark.y, landmark.z, landmark.visibility], dtype=np.float32)
    
    leftHLmk = np.zeros((21,3), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            leftHLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    rightHLmk = np.zeros((21,3), dtype=np.float32)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            rightHLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    return np.concatenate((faceLmk.flatten(), poseLmk.flatten(), leftHLmk.flatten(), rightHLmk.flatten()))

#------------------ Feb 03 2026 ------------------
#------------------ Velocity Check Function ------------------

# partialData is expected to be a 1D numpy array that represents a 1 FRAME feature (landmark coordinate).

# def checkVelocity(partialDataA, partialDataB):
#     velocityArr = partialDataB[1404:] - partialDataA[1404:]

#     # return np.max(np.abs(velocityArr)) > threshold
#     # needs better thresholding based on experimentation
#     return True if np.linalg.norm(velocityArr) > 0.44 else False

    # there's not much to update the function here
    # most values are close to 0.1 then jumps to 0.5
    # no improvements here unless running for a different algorithm

def checkVelocity(partialDataA, partialDataB):
    velocityArr = partialDataB - partialDataA
    return True if np.linalg.norm(velocityArr) > 0.44 else False

#------------------ Feb 18 2026 ------------------

#------------------ Landmark Extraction Solely for face ------------------
#------------------ 1404 landmarks ---------------------------------------
def extractFaceLandmarks(results):
    faceLmk = np.zeros((468,3), dtype=np.float32)
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            faceLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    return faceLmk


#------------------ Landmark Extraction Solely for torso, arms, hands ------------------------------
# not used right now, too many functions will be affected if I change the output format of this one, so I will just make a new one for now.
#------------------ 132 pose landmarks + 63 left hand landmarks + 63 right hand landmarks = 258 landmarks ------------------

def extractPoseLandmarks(results):
    poseLmk = np.zeros((33,4), dtype=np.float32)
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            poseLmk[i] = np.array([landmark.x, landmark.y, landmark.z, landmark.visibility], dtype=np.float32)
    
    leftHLmk = np.zeros((21,3), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            leftHLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    rightHLmk = np.zeros((21,3), dtype=np.float32)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            rightHLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    return np.concatenate((poseLmk.flatten(), leftHLmk.flatten(), rightHLmk.flatten()))


# comment later

def faceNormalization(faceLmk):

    nose_tip_index = 1  # Assuming the nose tip is at index 1
    centered_landmarks = faceLmk - faceLmk[nose_tip_index]  # Center the landmarks around the nose tip
    # max_distance = np.max(np.abs(centered_landmarks))  # Find the maximum distance from the center
    max_distance = np.max(np.linalg.norm(centered_landmarks, axis=1))  # Alternative: max Euclidean distance from the center
    if max_distance > 0:
        normalized_landmarks = centered_landmarks / max_distance  # Normalize to fit within a unit circle
    else:
        normalized_landmarks = centered_landmarks  # If all landmarks are the same, keep them as is

    # Left eyebrow landmarks: 70, 63, 105, 66, 107
    # Left eye landmarks: 33, 160, 158, 133, 153, 144
    left_brow_mean = normalized_landmarks[70:108][:, 1].mean()   # mean y of brow region
    left_eye_mean  = normalized_landmarks[33:134][:, 1].mean()   # mean y of eye region
    left_brow_raise = left_brow_mean - left_eye_mean  # relative raise

    # Right eyebrow landmarks: 300, 293, 334, 296, 336
    # Right eye landmarks: 263, 387, 385, 362, 380, 373
    right_brow_mean = normalized_landmarks[300:336][:, 1].mean() # mean y of brow region
    right_eye_mean  = normalized_landmarks[263:362][:, 1].mean() # mean y of eye region
    right_brow_raise = right_brow_mean - right_eye_mean  # relative raise

    left_brow_mean_pos  = normalized_landmarks[70:108].mean(axis=0)   # (3,) mean x,y,z of left brow
    right_brow_mean_pos = normalized_landmarks[300:336].mean(axis=0)  # (3,) mean x,y,z of right brow

    brow_distance = np.linalg.norm(left_brow_mean_pos - right_brow_mean_pos)  # scalar


    # Mouth corners
    left_corner  = normalized_landmarks[61]   # (3,)
    right_corner = normalized_landmarks[291]  # (3,)
    mouth_corner_diff = (left_corner - right_corner).astype(np.float32)  # (3,) captures asymmetry too

    mouth_open = normalized_landmarks[13][1]  - normalized_landmarks[14][1]
    mouth_width = normalized_landmarks[61][0]  - normalized_landmarks[291][0]

    extra_scalars = np.array([left_brow_raise, right_brow_raise, 
                           mouth_open, mouth_width, 
                           brow_distance], dtype=np.float32)  # (5,)

    extra = np.concatenate([extra_scalars, mouth_corner_diff])    # (5+3,) = (8,)

    return np.concatenate([normalized_landmarks.flatten(), extra]) # (1404+8,) = (1412,)