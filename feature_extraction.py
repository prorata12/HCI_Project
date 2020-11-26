import cv2
import mediapipe as mp
import numpy as np #Added
import file_read as fr #Added
import copy

# Link : https://google.github.io/mediapipe/solutions/face_mesh


def one_folder_landmarking(folder):

  for idx, file in enumerate(folder):
    # in folder, process pictures one by one
    # idx :  picture index
    # file : path of image file)
    
    print("Picture Path: ",file)
    image = cv2.imread(file)

    cv2.imshow("Original",image)
    cv2.waitKey(0)

    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    # 468 : number of facial landmarks in MediaPipe
    x_pos = np.zeros((len(folder),468))
    y_pos = np.zeros((len(folder),468))
    z_pos = np.zeros((len(folder),468))
    
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)

      print("Landmark (first sample)\n",face_landmarks.landmark[0])
      print("Number of Landmarks:",len(face_landmarks.landmark)) # 468 Landmarks

      for index,landmark in enumerate(face_landmarks.landmark):
        x_pos[idx][index] = landmark.x
        y_pos[idx][index] = landmark.y
        z_pos[idx][index] = landmark.z


      cv2.imshow("Annotated",annotated_image)
      cv2.waitKey(0)

      # You can save images using this
      #cv2.imwrite('./original_image' + str(idx) + '.png', image)
      #cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)

  face_mesh.close()
  x_pos[idx][index] = landmark.x
  y_pos[idx][index] = landmark.y
  z_pos[idx][index] = landmark.z
  return x_pos, y_pos, z_pos




if __name__ == "__main__":

  mp_drawing = mp.solutions.drawing_utils
  mp_face_mesh = mp.solutions.face_mesh

  # For static images:
  face_mesh = mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      min_detection_confidence=0.5)
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



  file_list = copy.deepcopy(fr.file_list)  # [emotion][0: train, 1: test, 2:label]

  one_folder_landmarking(file_list[0][0])
  one_folder_landmarking(file_list[0][1])
  # file_list[0][2] stores 0 (emotion label of pictures)

  one_folder_landmarking(file_list[1][0])
  one_folder_landmarking(file_list[1][1])
  # file_list[1][2] stores 1 (emotion label of pictures)




# =================================================================================================== #
# webcam is not tested yet. This code is raw version from mediapipe site.

# For webcam input:
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = face_mesh.process(image)

  # Draw the face mesh annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
  cv2.imshow('MediaPipe FaceMesh', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
face_mesh.close()
cap.release()

