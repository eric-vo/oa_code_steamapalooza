import numpy as np
import cv2 as cv

# Names of the images to be used (without file extension)
NAMES = ('angelina', 'anna', 'eric', 'veronica', 'sean', 'derek', 'trisha')

# Text parameters
TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2
TEXT_PADDING = 35

# Load ArUco dictionary
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

# Initialize camera
cap = cv.VideoCapture(0)

# Check if camera was opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect markers in the frame
    corners, ids, rejected = cv.aruco.detectMarkers(frame, aruco_dict)

    # If markers are detected
    if ids is not None:
        # Loop over each marker
        for i in range(len(ids)):
            # Get the ID of the marker
            id = ids[i][0]

            # If the ID corresponds to an image
            if id < len(NAMES):
                # Get the corners of the current marker
                corners_i = corners[i][0]

                # Load the image (PNG format)
                image = cv.imread(f'{NAMES[id]}.png')

                # Calculate the perspective transformation
                # between the marker and the image
                M = cv.getPerspectiveTransform(
                    np.array(  # Image corners
                        [[0, 0],
                         [image.shape[1], 0],
                         [image.shape[1], image.shape[0]],
                         [0, image.shape[0]]], dtype='float32'
                    ),
                    corners_i  # Marker corners
                )

                # Apply the perspective transformation
                # to the image to fit the marker
                warped_image = cv.warpPerspective(
                    image, M, (frame.shape[1], frame.shape[0])
                )

                # Create a mask
                marker_mask = np.zeros(frame.shape)

                # Fill the mask with the marker shape in white
                cv.fillConvexPoly(
                    marker_mask, corners_i.astype('int32'), (255, 255, 255)
                )

                # Replace the areas of the frame
                # highlighted by the mask with the warped image
                frame[marker_mask != 0] = warped_image[marker_mask != 0]

                # Get the center x and lowest y of the marker
                marker_center_x = int(np.mean(corners_i[:, 0]))
                marker_lowest_y = int(max(corners_i[:, 1]))

                # Get the name of the image and the size the text will have
                name = NAMES[id]
                text_size, baseline = cv.getTextSize(
                    name, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
                )

                # Get the color of the pixel
                # at the center of where the text will be
                text_bg_color = frame[
                    min(marker_lowest_y + TEXT_PADDING + text_size[1],
                        frame.shape[0] - 1),
                    marker_center_x
                ]

                # Find its inverse color
                text_inv_bg_color = tuple(
                    255 - int(value) for value in text_bg_color
                )

                # Draw the name of the image below the marker
                cv.putText(
                    frame, name,
                    (marker_center_x - text_size[0] // 2,
                     marker_lowest_y + TEXT_PADDING),
                    TEXT_FONT, TEXT_SCALE, text_inv_bg_color, TEXT_THICKNESS
                )

    # Display the resulting frame
    cv.imshow('frame', frame)

    # Exit if "q" is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release camera and close windows
cap.release()
cv.destroyAllWindows()
