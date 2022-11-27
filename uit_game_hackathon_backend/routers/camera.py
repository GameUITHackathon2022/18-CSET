from logging import getLogger
from fastapi import APIRouter, Form, WebSocket, WebSocketDisconnect

from src.rcode import rcode
from src.models.camera.camera import Camera
from models.camera.camera_multi import VideoStreamController

router = APIRouter()

logger = getLogger("app")


@router.get('/camera')
def get_list_user_camera(
    user_id: str
):
    err, cameras = Camera.get_user_cameras(user_id=user_id)

    if err:
        return rcode(err)
    
    return {
        **rcode(1000),
        'cameras': cameras
    }

@router.post('/camera')
def post_new_camera(
    name: str = Form(...),
    user_name: str = Form(...),
    password: str = Form(...),
    protocol: str = Form(...),
    is_hidden: str = Form(...),
    user_id: str = Form(...),
):
    
    err, camera = Camera(
                    name=name,
                    user_name=user_name,
                    password=password,
                    protocol=protocol,
                    is_hidden=is_hidden,
                    user_id=user_id
                    ) 
    
    if err:
        return rcode(err)
    
    return rcode(1000)

@router.get('/streaming')
async def streaming():
    pass


from threading import Thread

video_stream_controller = VideoStreamController()
checkin_thread = Thread(target=video_stream_controller.start,args=())
checkin_thread.daemon = True
checkin_thread.start()

@router.websocket("/ws_streaming")
async def socket_streaming(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            mess = await websocket.receive_text()
            image = video_stream_controller.show_frame()
            if image is not None:
                await websocket.send_bytes(image)
    except WebSocketDisconnect:
        logger.error("Client disconnected")
    finally:
        websocket.close()