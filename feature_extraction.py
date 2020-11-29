import cv2
import mediapipe as mp
import numpy as np #Added
import file_read as fr #Added
import copy
import os

# Link : https://google.github.io/mediapipe/solutions/face_mesh


def one_folder_landmarking(folder,index):
  mp_drawing = mp.solutions.drawing_utils
  mp_face_mesh = mp.solutions.face_mesh

  # For static images:
  face_mesh = mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      min_detection_confidence=0.5)
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # You can save images using this
  original_path = 'D:/Desktop_D/HCI_Project/' + str(index) +'/original/'
  annotated_path = 'D:/Desktop_D/HCI_Project/' + str(index) +'/annotated/'

  if not os.path.exists(original_path):
    os.makedirs(original_path)
  if not os.path.exists(annotated_path):
    os.makedirs(annotated_path)

  errorcount = 0
  x_pos = []
  y_pos = []
  z_pos = []
  for idx, file in enumerate(folder):
    # in folder, process pictures one by one
    # idx :  picture index
    # file : path of image file)
    
    image = cv2.imread(file)
    #cv2.imshow("Original",image)
    #cv2.waitKey(0)

    # Convert the BGR image to RGB before processing.
    try:
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except:
      errorcount += 1
      continue
    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    # 468 : number of facial landmarks in MediaPipe
    x_pos_one_picture = np.zeros(468)
    y_pos_one_picture = np.zeros(468)
    z_pos_one_picture = np.zeros(468)
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)

      #print("Landmark (first sample)\n",face_landmarks.landmark[0])
      #print("Number of Landmarks:",len(face_landmarks.landmark)) # 468 Landmarks

      for index,landmark in enumerate(face_landmarks.landmark):
        x_pos_one_picture[index] = landmark.x
        y_pos_one_picture[index] = landmark.y
        z_pos_one_picture[index] = landmark.z


    #cv2.imshow("Annotated",annotated_image)
    #cv2.waitKey(0)
    x_pos.append(x_pos_one_picture)
    y_pos.append(y_pos_one_picture)
    z_pos.append(z_pos_one_picture)

    cv2.imwrite(original_path + str(idx) + '.png', image)
    cv2.imwrite(annotated_path + str(idx) + '.png', annotated_image)
  face_mesh.close()

  print('errorcount = ',errorcount)
  return x_pos, y_pos, z_pos




if __name__ == "__main__":

  num_of_emotions = len(fr.train_pictures)
  #print(num_of_emotions)
  x_pos_train = [[] for i in range(num_of_emotions)]
  y_pos_train = [[] for i in range(num_of_emotions)]
  z_pos_train = [[] for i in range(num_of_emotions)]
  #print(x_pos_train[0])
  x_pos_test = [[] for i in range(num_of_emotions)]
  y_pos_test = [[] for i in range(num_of_emotions)]
  z_pos_test = [[] for i in range(num_of_emotions)]

  emotion_num = len(fr.train_paths)


  for idx,folder in enumerate(fr.train_pictures):
    print('train folder num=',idx)
    x,y,z = one_folder_landmarking(folder,idx)
    x_pos_train[idx] = copy.deepcopy(x)
    y_pos_train[idx] = copy.deepcopy(y)
    z_pos_train[idx] = copy.deepcopy(z)
  
  for idx,folder in enumerate(fr.test_pictures):
    print('train folder num=',idx)
    x,y,z = one_folder_landmarking(folder,idx)
    x_pos_text[idx] = copy.deepcopy(x)
    y_pos_text[idx] = copy.deepcopy(y)
    z_pos_text[idx] = copy.deepcopy(z)
  
  x_pos_train = np.array(x_pos_train)
  y_pos_train = np.array(y_pos_train)
  z_pos_train = np.array(z_pos_train)
  
  x_pos_test = np.array(x_pos_test)
  y_pos_test = np.array(y_pos_test)
  z_pos_test = np.array(z_pos_test)

  np.save('D:/Desktop_D/HCI_Project/x_pos_train',x_pos_train)
  np.save('D:/Desktop_D/HCI_Project/y_pos_train',y_pos_train)
  np.save('D:/Desktop_D/HCI_Project/z_pos_train',z_pos_train)
  np.save('D:/Desktop_D/HCI_Project/x_pos_test',x_pos_test)
  np.save('D:/Desktop_D/HCI_Project/y_pos_test',y_pos_test)
  np.save('D:/Desktop_D/HCI_Project/z_pos_test',z_pos_test)



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

