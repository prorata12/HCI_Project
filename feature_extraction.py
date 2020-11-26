# https://google.github.io/mediapipe/solutions/face_mesh.html
import cv2
import mediapipe as mp
import numpy as np #Added
import file_read as fr #Added
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#### Added ####
f1 = "D:/Desktop_D/face1.jpg"
# f2 = "D:/Desktop_D/face2.jpg"
# f3 = "D:/Desktop_D/face3.jpg"
# f4 = "D:/Desktop_D/HCI_Project/archive/train/happy/Training_10019449.jpg"
# file_list = [f1,f2,f3,f4]
file_list = fr.file_list[0][0]
print(file_list)
#### Added end ####

for idx, file in enumerate(file_list):
  print(file)
  image = cv2.imread(file)

  #### Added ####
  x_size = len(image[0])
  y_size = len(image)
  print(len(image))
  #image = cv2.line(image,(x_size,y_size),(150,150),(0,0,255),2)
  cv2.imshow("Original",image)
  cv2.waitKey(0)
  #### Added End ####

  # Convert the BGR image to RGB before processing.

  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print and draw face mesh landmarks on the image.
  if not results.multi_face_landmarks:
    continue
  annotated_image = image.copy()

  #### Added ####
  x_pos = np.zeros((len(file_list),468))
  y_pos = np.zeros((len(file_list),468))
  z_pos = np.zeros((len(file_list),468))
  #### Added End ####
  
  for pic_num, face_landmarks in enumerate(results.multi_face_landmarks): #Modified
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
    #### Added ####
    print("Landmarks",face_landmarks.landmark[0],"End")
    print("Landmarks",len(face_landmarks.landmark),"End") # 468 Landmarks
    # print("Landmarks",face_landmarks.landmark[0].x,"End")
    # print("Landmarks",face_landmarks.landmark[0].y,"End")
    # print("Landmarks",face_landmarks.landmark[0].z,"End")

    for index,landmark in enumerate(face_landmarks.landmark):
      x_pos[idx][index] = landmark.x
      y_pos[idx][index] = landmark.y
      z_pos[idx][index] = landmark.z
    #### Added End ####

    #### Added ####
    cv2.imshow("Annotated",annotated_image)
    cv2.waitKey(0)
    #cv2.imwrite('./original_image' + str(idx) + '.png', image)
    #cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)
    #### Added End ####
face_mesh.close()




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


####
#ML PART
####

x_pos[idx][index] = landmark.x
y_pos[idx][index] = landmark.y
z_pos[idx][index] = landmark.z
