import os
import logging
import datetime
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.utils import rlogger
from src.utils.configs import get_config
# from routers import streaming, checkin
from routers import  user
from routers import misson
# from models.camera.camera_multi import VideoStreamController

now = datetime.datetime.now()

# load config
config = get_config().app

SERVICE_IP = config.service_ip
SERVICE_PORT = config.service_port
LOG_PATH = config.log.dir

# create folder structure
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

app = FastAPI()

#CORS
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create logger
log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s" " %(funcName)s(%(lineno)d) %(message)s"
)
log_handler = rlogger.BiggerRotatingFileHandler(
    "ali",
    LOG_PATH,
    mode="a",
    maxBytes=2 * 1024 * 1024,
    backupCount=200,
    encoding=None,
    delay=0,
)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

logger.info("INIT LOGGER SUCCESSED")


# print app info
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("API READY")

app.include_router(user.router, prefix="/user")
app.include_router(misson.router, prefix="/mission")
# app.include_router(checkin.router, prefix="/checkin")

@app.get("/")
def healthy():
    return "Ok"



# socket_server_thread = Thread(target=video_stream_controller.streaming,args=())
# socket_server_thread.daemon = True
# socket_server_thread.start()
# video_stream_controller.start()
