import streamlit as st
import cv2
import numpy as np
import dlib
# Set the title and page layout
from PIL import Image

def apply_google1(image):
    st.write("Google Apply Function 1 Executed!")
    goggle_name="goggle1.jpg"
    sunglasses(image,goggle_name)

def apply_google2(image):
    st.write("Google Apply Function 2 Executed!")
    goggle_name="goggle3.jpg"
    sunglasses(image,goggle_name)

def apply_google3(image):
    st.write("Google Apply Function 3 Executed!")
    goggle_name="goggle4.jpg"
    sunglasses(image,goggle_name)

def apply_google4(image):
    st.write("Google Apply Function 4 Executed!")
    goggle_name="goggle2.jpg"
    sunglasses(image,goggle_name)

def sunglasses(image,goggle_name):
    
    finalimg = image.copy()
    # Load the eye cascade classifier
    cascade_path = "frontalEyes35x16.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(image)
    print(eyes)

    # Get the coordinates of the first detected eye
    if len(eyes) > 0:
        eye_x, eye_y, eye_w, eye_h = eyes[0]

        # Draw a rectangle around the detected eye
        img = cv2.rectangle(image, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 255), 2)
        print("Eye detected:")
       
        # Load and resize the glasses filter image
        glasses_filter = cv2.imread(goggle_name, cv2.IMREAD_UNCHANGED)
        glasses_filter_resized = cv2.resize(glasses_filter, (eye_w + 50, eye_h + 55))

        # Overlay the glasses filter onto the image
        for i in range(glasses_filter_resized.shape[0]):
            for j in range(glasses_filter_resized.shape[1]):
                if glasses_filter_resized[i, j, 3] > 0:
                    finalimg[eye_y + i - 22, eye_x + j - 23, :] = glasses_filter_resized[i, j, :-1]
        st.image(finalimg, caption="Your Makeup Results")


def lipstick(image, color):
    # Load the Haar Cascade classifier for face detection
   

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    finalimg = image.copy()

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Load the shape predictor model
    model_path = "models/shape_predictor_68_face_landmarks.dat" 
    predictor = dlib.shape_predictor(model_path)

    landmarkspoints = []

    for (x, y, w, h) in faces:
        face_rect = dlib.rectangle(x, y, x + w, y + h)

        # Predict landmarks for the detected face
        landmarks = predictor(imgGray, face_rect)

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarkspoints.append([x, y])

    #st.image(finalimgout, caption="Your Makeup Results", use_column_width=400, width=300)      
    #code for lips detection, creating a mask, and changing lip color
    landmarkspoints = np.array(landmarkspoints)
    lipmask = np.zeros_like(finalimg)  # Use 'finalimg' instead of 'image'
    lipimg = cv2.fillPoly(lipmask, [landmarkspoints[48:60]], (255, 255, 255))

    # Change the lip color
    lipimgcolor = np.zeros_like(lipimg)
    b = color[0][0][0]
    g = color[0][0][1]
    r = color[0][0][2]
    print(b,g,r)
    lipimgcolor[:] = b, g, r

    lipimgcolor = cv2.bitwise_and(lipimg, lipimgcolor)

    lipimgcolor = cv2.GaussianBlur(lipimgcolor, (7, 7), 3)
    finalimgout = cv2.addWeighted(finalimg, 1, lipimgcolor, 0.7, 0)

    st.image(finalimgout, caption="Your Makeup Results", use_column_width=400, width=300)

def eyeLenses(img):
    #img = cv2.imread("girl2.jpg")

    ori = img.copy()

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()

    # Add a predictor
    predictor = dlib.shape_predictor("models/shape_predictor_70_face_landmarks.dat")
    
    red_value = st.sidebar.slider("Red", 0, 255, 127)
    green_value = st.sidebar.slider("Green", 0, 255, 127)
    blue_value = st.sidebar.slider("Blue", 0, 255, 127)

    # Create an image based on the selected RGB values
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :, 0] = red_value
    color_image[:, :, 1] = green_value
    color_image[:, :, 2] = blue_value

    # Display the generated image in the Streamlit sidebar
    st.sidebar.image(color_image, caption="Generated Image", use_column_width=True)
    # Start to detect faces in the image
    faces = detector(grayimg)

    landmarkpts = []

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Draw the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Predict the face components
        landmarks = predictor(grayimg, face)
        for n in range(68, 70):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 2, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(n), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (222, 123, 0), 2)
            landmarkpts.append([x, y])

        # Preparing the mask
        mask = np.zeros_like(img)
        print(landmarkpts)
        lefteye = cv2.circle(mask, (landmarkpts[0][0], landmarkpts[0][1]), 5, (255, 255, 255), cv2.FILLED)
        righteye = cv2.circle(mask, (landmarkpts[1][0], landmarkpts[1][1]), 5, (255, 255, 255), cv2.FILLED)
    
        b, g, r = red_value,green_value,blue_value
        eyecolor = np.zeros_like(mask)
        eyecolor[:,:] = b, g, r
       
        eyecolormask = cv2.bitwise_and(mask, eyecolor)
    
        eyecolormask = cv2.GaussianBlur(eyecolormask, (7, 7), 10)
        

        finalimg = cv2.addWeighted(ori, 1, eyecolormask, 0.4, 0)
        st.image(finalimg, caption="Your Makeup Results", use_column_width=400, width=300)



st.set_page_config(
    page_title="Design Virtual Makeup",
    layout="wide",
)
st.subheader("Where Your Imagination Meets the Mirror")
# Define the sidebar
st.sidebar.title("Virtual Makeup")

def eyeLiner(image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        finalimg = image.copy()
        # Convert the image to grayscale for face detection
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("busfo",imgGray)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        red_value = st.sidebar.slider("Red", 0, 255, 127)
        green_value = st.sidebar.slider("Green", 0, 255, 127)
        blue_value = st.sidebar.slider("Blue", 0, 255, 127)

        # Create an image based on the selected RGB values
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :, 0] = red_value
        color_image[:, :, 1] = green_value
        color_image[:, :, 2] = blue_value

        # Display the generated image in the Streamlit sidebar
        st.sidebar.image(color_image, caption="Generated Image", use_column_width=True)
        # Load the shape predictor model
        model_path = "models/shape_predictor_68_face_landmarks.dat" 
        predictor = dlib.shape_predictor(model_path)

        # Initialize a list to store landmark points
        landmarkspoints = []

        for (x, y, w, h) in faces:
            # Get the face rectangle
            face_rect = dlib.rectangle(x, y, x + w, y + h)

            # Predict landmarks for the detected face
            landmarks = predictor(imgGray, face_rect)

            # Extract landmark points for the face
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarkspoints.append([x, y])
                cv2.circle(image, (x, y), 3, (255, 255, 0), cv2.FILLED)
                cv2.putText(image,str(n),(x+1,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,0,0),1)

        #cv2.imshow("hakc",image)

        right_eye_counter_top = np.array(landmarkspoints[36:40])
        right_eye_counter_top[:,1] = right_eye_counter_top[:,1]-3
        right_eye_counter_bottom = np.array(landmarkspoints[39:42])
        right_eye_counter_bottom = np.append(right_eye_counter_bottom,[np.array(landmarkspoints[36])],axis=0)

        left_eye_counter_top = np.array(landmarkspoints[42:46])
        left_eye_counter_top[:,1] = left_eye_counter_top[:,1]-3
        left_eye_counter_bottom = np.array(landmarkspoints[45:48])
        left_eye_counter_bottom = np.append(left_eye_counter_bottom,[np.array(landmarkspoints[42])],axis=0)

        finalimg = cv2.polylines(finalimg,[right_eye_counter_top],False,(blue_value,green_value,red_value),2)
        #finalimg = cv2.polylines(finalimg,[right_eye_counter_bottom],False,(0,0,0),2)
        finalimg = cv2.polylines(finalimg,[left_eye_counter_top],False,(blue_value,green_value,red_value),2)
        #finalimg = cv2.polylines(finalimg,[left_eye_counter_bottom],False,(0,0,0),2)
        
        #cv2.imshow("gfgdfsd",finalimg[:,:,::-1])
        st.image(finalimg[:,:,::-1], caption="Your Makeup Results")




    # Custom CSS for setting the background image
st.title("Makeup Magic - Virtual Makeup Try-On")
# Create a dropdown menu for selecting makeup options
selected_option = st.sidebar.selectbox("Select Makeup Option:", ["Lipsticks", "Eyelenses", "Goggles", "Eyeliners"])

if selected_option == "Lipsticks":
    # Create a Streamlit sidebar with three sliders for RGB values
    red_value = st.sidebar.slider("Red", 0, 255, 127)
    green_value = st.sidebar.slider("Green", 0, 255, 127)
    blue_value = st.sidebar.slider("Blue", 0, 255, 127)

    # Create an image based on the selected RGB values
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :, 0] = red_value
    color_image[:, :, 1] = green_value
    color_image[:, :, 2] = blue_value

    # Display the generated image in the Streamlit sidebar
    st.sidebar.image(color_image, caption="Generated Image", use_column_width=True)

    # Optionally, display the RGB values below the image
    st.sidebar.write(f"Red: {red_value}, Green: {green_value}, Blue: {blue_value}")

    # Upload User's Photo
    st.sidebar.title("Upload Your Photo")
    user_image = st.sidebar.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])

    #if user_image is not None:
     #   st.sidebar.image(user_image, caption="Uploaded Photo", use_column_width=True)

    if user_image is not None:
        # Open the uploaded image using PIL
        pil_image = Image.open(user_image)
        # Convert the PIL image to a NumPy array
        numpy_image = np.array(pil_image)
        print("type of image")
        print(type(user_image))
        
        lipstick(numpy_image,color_image)

elif selected_option == "Eyelenses":
    st.subheader("Eye lenses")
    # Upload User's Photo
    st.sidebar.title("Upload Your Photo")
    user_image = st.sidebar.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])
    
    if user_image is not None:
        pil_image = Image.open(user_image)
        numpy_image = np.array(pil_image)
        eyeLenses(numpy_image)

elif selected_option == "Goggles":
    st.subheader("Goggles Options")
    # Upload User's Photo
    user_image = st.sidebar.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])
    if user_image is not None:
       pil_image = Image.open(user_image)
       numpy_image = np.array(pil_image)
    #Display three images in a row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("goggle1.jpg", caption="Image 1", use_column_width=True)
        if st.button("Apply Google 1"):
               apply_google1(numpy_image)

    with col2:
        st.image("goggle3.jpg", caption="Image 2", use_column_width=True)
        if st.button("Apply Google 2"):
            if user_image is not None:
               apply_google2(numpy_image)

    with col3:
        st.image("image1.jpg", caption="Image 3", use_column_width=True)
        if st.button("Apply Google 3"):
            if user_image is not None:
               apply_google3(numpy_image)
    
    with col1:
        st.image("goggle2.jpg", caption="Image 1", use_column_width=True)
        if st.button("Apply Google 4"):
               apply_google4(numpy_image)

elif selected_option == "Eyeliners":
    st.sidebar.title("Upload Your Photo")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        #st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
        eyeLiner(image)
    





