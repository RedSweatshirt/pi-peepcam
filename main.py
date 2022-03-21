import numpy as np
import imutils
import cv2 as cv
import time
import PIL
from PIL import Image

# hardware specific
import pigpio                           # gpio
import explorerhat as eh                # pimoroni explorer hat pro
from picamera import PiCamera           # arducam
from picamera.array import PiRGBArray   # get RGB array
import ST7789                           # st7789 240x240 screen

pi = pigpio.pi()
camera = PiCamera()

res = (1280, 720)
camera.resolution = res
camera.framerate = 30

disp = ST7789.ST7789(
    port=0,
    cs=0,
    dc=15,
    backlight=None,
    mode=3,
    rotation=270,
    spi_speed_hz=90 * 1000 * 1000,
)

disp.begin()
WIDTH = disp.width
HEIGHT = disp.height

blk = 18    # backlight
pi.set_mode(blk, pigpio.OUTPUT)
brightness = 100
pi.hardware_PWM(blk, 300, brightness * 10000)   # pin, hz, duty cycle as fraction out of 1MHz

on_motion = True    # means screen only on for motion
min_area = 100      # area of the motion for detection
screen_time = 30     # time that the screen will be on, in seconds
screen_start = 0    # time that the screen was turned on, in ms

detected = False
is_screen_on = False


def size_down(img):
    """
    Takes in an image and creates a downscaled image of size 240x240
    :param img: numpy ndarray, image. dimensions are (height, width, channels)
    :return: a downscaled version of img, to size "size"
    """
    height, width, channels = img.shape
    # first crop the image so that resize works
    if height > width:
        diff = height - width
        img = img[diff//2:height-(diff//2), :, :]
    if width > height:
        diff = width - height
        img = img[:, diff//2:width-(diff//2), :]

    return cv.resize(img, (HEIGHT, WIDTH), interpolation=cv.INTER_AREA)


def set_brightness(level):
    """
    sets the screen brightness
    :param level: int between 0 and 100 inclusive
    :return: nothing
    """
    pi.hardware_PWM(blk, 300, level * 10000)


def screen_on():
    """
    Turns the screen on
    :return: nothing
    """
    global is_screen_on
    is_screen_on = True
    set_brightness(100)


def screen_off():
    """
    Turns the screen off
    :return: nothing
    """
    global is_screen_on
    is_screen_on = False
    image = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=(255, 255, 255))
    disp.display(image)
    set_brightness(0)


def display_frame(frame):
    """
    Displays a frame on the st7789 240x240 screen
    :param frame: ndarray image. dimensions are (height, width, channels)
    :return: nothing
    """
    if frame is None:
        return
    frame = size_down(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    disp.display(image)


def screen_update(frame=None, detected=False):
    """
    Updates the screen
    :param frame: ndarray image. dimensions are (height, width, channels)
    :param detected: boolean, says whether motion was detected or not
    :return: nothing
    """
    global screen_start
    global on_motion
    global is_screen_on

    if on_motion:
        if detected:
            screen_start = time.time()
            screen_on()
        elif time.time() - screen_start > screen_time:
            screen_off()
    elif not is_screen_on:
        screen_on()

    if frame is not None and is_screen_on:
        display_frame(frame)


def eh_update():
    """
    Updates the explorer hat inputs
    :return: nothing
    """
    global on_motion
    if eh.touch.one.is_pressed():
        on_motion = False
    if eh.touch.two.is_pressed():
        on_motion = True


if __name__ == "__main__":
    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(camera, size=res)

    # Wait a certain number of seconds to allow the camera time to warmup
    time.sleep(0.1)

    # Initialize the first frame of the video stream
    first_frame = None

    # Capture frames continuously from the camera
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

        if frame is None:
            print("oh no")
            continue

        eh_update()

        # Grab the raw NumPy array representing the image
        frame = frame.array

        if on_motion:
            detected = False
            # resize the frame, convert it to grayscale, and blur it
            small_frame = imutils.resize(frame, width=500)
            gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (21, 21), 0)

            # If first frame, we need to initialize it.
            if first_frame is None:
                first_frame = gray
                # Clear the stream in preparation for the next frame
                raw_capture.truncate(0)
                # Go to top of for loop
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frame_delta = cv.absdiff(first_frame, gray)
            thresh = cv.threshold(frame_delta, 100, 255, cv.THRESH_BINARY)[1]
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv.dilate(thresh, None, iterations=2)
            conts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            conts = imutils.grab_contours(conts)
            # loop over the contours
            for c in conts:
                # if the contour is too small, ignore it
                if cv.contourArea(c) < min_area:
                    continue

                # if you get here, means motion detected
                detected = True
                break

            if detected:
                screen_update(frame, detected)
                first_frame = gray
            else:
                screen_update(frame)

        else:   # screen always on, no motion detection
            screen_update(frame)

        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)

