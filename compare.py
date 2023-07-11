import face_recognition
import os

picture_of_me = face_recognition.load_image_file("IMG2/benazir-bhutto2.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

for f in os.listdir("IMG1"):
    print(f'Checking: {f}')
    unknown_picture = face_recognition.load_image_file(f"IMG1/{f}")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

    # Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

    if results[0] == True:
        print(f"It's a picture of me! {f}")
    else:
        print(f"It's not a picture of me! {f}")
