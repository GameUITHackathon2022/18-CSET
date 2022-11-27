from fastapi import APIRouter, Form

from src.models.mission.mission import Mission
from src.models.user.user import User
from src.models.joiner.joiner import Joiner
from src.rcode import rcode

router = APIRouter()


@router.get('/join_task')
def user_join_task(
    user_id: int ,
    mission_id: int 
):
    joiner = Joiner(userid=user_id,mission_id=mission_id)
    err, _ = Joiner.insert(joiner)

    if err:
        return rcode(err)

    return rcode(1000)

@router.get('/pending_task')
def get_pending_task_task(
    user_id: int
):
    err, joiners = Joiner.get_all_by_user_id(user_id=user_id)

    if err:
        return rcode(err)

    list_mission = []

    for joiner in joiners:
        err, mission = Mission.get_all_mission_by_id(mission_id=joiner.mission_id)

        if err != "NotFound" and err:
            return rcode(err)
        elif err == "NotFound":
            continue
        
        list_mission.append(mission)

    return {
        **rcode(1000),
        'missions': list_mission
    }

@router.get('/completed')
def report_completed(
    user_id: int,
    mission_id: int
):  
    # err, joiner = Joiner.get_all_by_user_id(user_id=user._id)
    
    err, task = Mission.get_all_mission_by_id(mission_id)
    if err:
        return rcode(err)
    
    if task.is_done_model:
        err,_ = User.update_score(user_id=user_id,score=task.mission_score)

        if err:
            return rcode(err)

        err,_ = Mission.update_status_mission(status=1,mission_id=mission_id)

        if err:
            return rcode(err)

        return rcode(1000)

    else:
        return rcode(1402)

@router.get("/user_mission")
def get_mission_by_user_id(
    user_id: int,
):
    err, joiners = Joiner.get_all_by_user_id(user_id=user_id)

    if err:
        return rcode(err)
    
    list_mission = []

    for joiner in joiners:
        err, mission = Mission.get_all_mission_by_id(joiner.mission_id)

        if err:
            return rcode(err)
        
        list_mission.append(mission)

    return {
        **rcode(1000),
        'missions': list_mission
    }


@router.get("/all_task_pending_of_user")
def get_all_task_pending_of_user(
    user_id: int,
):
    err, missions = Joiner.get_all_other_by_mission_id(user_id=user_id)

    if err:
        return rcode(err)


    return {
        **rcode(1000),
        'missions': missions
    }

import threading
from threading import Thread
import cv2
import requests
import asyncio
import concurrent.futures

def run(camera_id):
        cap = cv2.VideoCapture('2694495679494239092.mp4')
        count = 0
        i = 0
        while(cap.isOpened()):
        # Capture each frame
            ret, frame = cap.read()
            if ret == True:
                # cv2.imshow('Frame', frame)
                data = cv2.imencode('.jpg', frame)[1]
                res = requests.post(url="https://aiclub.uit.edu.vn/hackathon/2022/yolov5/predict_binary",
                        files=dict(binary_file=data)
                                )
                res = res.json()
                if res['code'] != '1000':
                    return rcode("SQLExecuteError")

                predicts = res['predicts']
                if i % 100 == 0:
                    cv2.imwrite('test.jpg',frame)
                err, mission = Mission.check_submition(camera_id)

                # if err:
                #     return rcode(err)
                
                if mission is None and len(predicts) > 0:
                    mission = Mission(_id=0,camera_id=camera_id,img_url='a',mission_score=100,location_desc='NVH SV, Thu Duc, TPHCM',mission_time='1',is_done=0,is_done_model=0)
                    err, _ = Mission.insert(mission)
                    
                    if err:
                        return rcode(err)
                    continue

                if len(predicts) == 0 and mission is not None:
                    count += 1
                
                if count >= 5 and len(predicts) == 0 and mission is not None:
                    Mission.update_model_status_mission(1,mission._id)
                    return rcode(1000)
                i += 1
import multiprocessing
@router.get("/run_video_demo")
def run_video_demo(
    camera_id: int,
):
    with multiprocessing.Pool(1) as p:
        result = p.map(run, [camera_id])
    return rcode(1000)

@router.get("/done")
def get_all_done(
    user_id: int
):
    err, mission = Mission.get_all_done(user_id)

    if err:
        return rcode(err)
    
    return {
        **rcode(1000),
        'missions': mission
    }




