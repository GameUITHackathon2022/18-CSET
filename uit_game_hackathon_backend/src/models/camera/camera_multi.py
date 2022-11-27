# import cv2
# import time
# import asyncio
# from logging import getLogger
# from imutils.video import VideoStream


# from src.models.camera.base_camera import BaseCamera
# from src.utils.configs import get_config
# from src.utils.image_handler import ImageHandler

# logger = getLogger("app")
# app_config = get_config()

# class Camera(BaseCamera):
#     def __init__(self, url):
#         super().__init__()
#         self.url = url

#     # over-wride of BaseCamera class frames method
#     @staticmethod
#     def frames():
#         camera = VideoStream(0).start()
#         while True:

#             img = camera.read()
#             # Frame size (1944, 2592, 3)
#             if img is not None:

#                 # Resize Image
#                 img = ImageHandler.resize(img, ratio=0.5)

#                 # Split frame to 4 frame
#                 frames = ImageHandler.split_image_into_4(img)

#                 #Add detect here
#                 err, new_frames = detect(frames)
#                 if err:
#                     logger.error(f"Calling Detector err with {err}")
#                     continue

#                 # Concat frame
#                 img = ImageHandler.concat_4_image(new_frames)

#                 # encode as a jpeg image and return it
#                 yield cv2.imencode(".jpg", img)[1].tobytes()
#             else:
#                 exit(0)
#                 logger.error("Streaming frame err")
#             # time.sleep(1)
            
# from threading import Thread
# import cv2, time, socket
# import threading

# #Init Socket Setting
# socket_config = app_config.socket
# BUFF_SIZE = socket_config.buff_size
# server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# socket_address = (socket_config.host_ip,socket_config.port)
# server_socket.bind(socket_address)
# server_socket.listen()


# class VideoStreamWidget(object):
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         # self.camera = VideoStream(src).start()
#         # Start the thread to read frames from the video stream
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
#         self.frame = None

#     def update(self):
#         # Read the next frame from the stream in a different thread
#         # while True:
#         #     self.frame = self.camera.read()
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()
#             time.sleep(1/30)
    
#     def get_frame_bytes(self):
#         # Display frames in main program
#         if self.frame is not None and self.status:
#             # Resize Image
#             try: 
#                 self.frame = ImageHandler.resize(self.frame, ratio=0.5)

#                 # Split frame to 4 frame
#                 frames = ImageHandler.split_image_into_4(self.frame)

#                 err, new_frames = detect(frames)
#                 if err:
#                     logger.error(f"Calling Detector err with {err}")

#                 # Concat frame
#                 self.frame = ImageHandler.concat_4_image(new_frames)

#                 # encode as a jpeg image and return it
#                 # frame =  cv2.imencode(".jpg", self.frame)[1].tostring()
#                 return self.frame
#             except Exception as e:
#                 return None
#         return None
        
# class VideoStreamController:
#     def __init__(self,url):
#         self.video_stream = VideoStreamWidget(url)   
#         # self.thread = Thread(target=self.streaming, args=())
#         # self.thread.daemon = True
#         # self.thread.start()
#         self.frame_bytes = None
        
#     # def streaming(self):
#     #     while True:
#     #         client_socket,addr =  server_socket.accept()
#     #         thread = threading.Thread(target=self.show_frame, args=(addr,client_socket))
#     #         thread.start()
#     def show_frame(self):
#         if self.frame_bytes is not None:
#             return None
#         return ImageHandler.image_to_base64( self.frame_bytes)
    
    
#     def start(self):
#         while True:
#             frame_bytes = self.video_stream.get_frame_bytes()
#             if frame_bytes is not None:
#                 self.frame_bytes = frame_bytes
            