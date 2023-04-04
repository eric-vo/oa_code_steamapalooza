import numpy as np
import cv2

# Names of the images to be used (without file extension)
NAMES = ("angelina", "anna", "eric", "veronica", "sean", "derek", "trisha")

# Text parameters
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2
TEXT_PADDING = 35

# Load ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    # Get frame
    frame = cap.read()[1]

    # Detect markers in the frame
    corners, ids = cv2.aruco.detectMarkers(frame, aruco_dict)[:2]

    # If markers are detected
    if ids is not None:
        # Loop over each marker
        for i in range(len(ids)):
            # Get the ID of the marker
            marker_id = ids[i][0]

            # If the ID corresponds to an image
            if marker_id < len(NAMES):
                # Get the corners of the marker
                marker_corners = corners[i][0]

                # Load the image (PNG format)
                image = cv2.imread(f"{NAMES[marker_id]}.png")

                # Calculate the perspective transformation
                # between the marker and the image
                homography = cv2.findHomography(
                    np.array(  # Image corners
                        [[0, 0],
                         [image.shape[1], 0],
                         [image.shape[1], image.shape[0]],
                         [0, image.shape[0]]]
                    ),
                    marker_corners  # Marker corners
                )[0]

                # Apply the perspective transformation
                # to the image to fit the marker
                warped_image = cv2.warpPerspective(
                    image, homography, (frame.shape[1], frame.shape[0])
                )

                # Create a mask
                mask = np.zeros(frame.shape)

                # Fill the mask with the marker shape in white
                cv2.fillConvexPoly(
                    mask, marker_corners.astype("int32"), (255, 255, 255)
                )

                # Replace the areas of the frame
                # highlighted by the mask with the warped image
                frame[mask != 0] = warped_image[mask != 0]

                # Get the center x and lowest y of the marker
                marker_center_x = int(np.mean(marker_corners[:, 0]))
                marker_lowest_y = int(max(marker_corners[:, 1]))

                # Get the name of the image and the size the text will have
                name = NAMES[marker_id]
                text_size = cv2.getTextSize(
                    name, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
                )[0]

                # Get the color of the pixel
                # at the center of where the text will be
                pixel_color = frame[
                    min(marker_lowest_y + TEXT_PADDING + text_size[1],
                        frame.shape[0] - 1),
                    marker_center_x
                ]

                # Find its inverse color
                inverse_color = tuple(
                    255 - int(value) for value in pixel_color
                )

                # Draw the name of the image below the marker
                cv2.putText(
                    frame, name,
                    (marker_center_x - text_size[0] // 2,
                     marker_lowest_y + TEXT_PADDING),
                    TEXT_FONT, TEXT_SCALE, inverse_color, TEXT_THICKNESS
                )

    # Display the resulting frame
    cv2.imshow("Camera", frame)

    # Exit if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
