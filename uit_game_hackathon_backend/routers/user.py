from fastapi import APIRouter, Form
from models.gift.gift import Gift

from src.models.user.user import User
from src.rcode import rcode

router = APIRouter()


@router.get("/")
def get_user_info(
    user_name: str,
    password: str,
):
    err, user = User.get(user_name=user_name,password=password)

    if err:
        return rcode(err)
    
    return {
        **rcode(1000),
        'user': user
    }

@router.post("/register")
def register(
    user_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):  
    user = User(name=user_name,email=email,password=password)
    err, user = User.insert(user)
    
    if err:
        return rcode(err)
    
    return {
        **rcode(1000),
        'user': user
        }

@router.get("/list_user")
def get_list_user(
    top: int = 20
):
    err, users = User.get_all(top)

    if err:
        return rcode(err)
    
    return {
        **rcode(1000),
        'users':users
    }

@router.get("/all_avail")
def get_all_avail():
    
    err, gifts = Gift.get_all_avail()

    if err:
        return rcode(err)
    
    return {
        **rcode(1000),
        'gifts': gifts
    }

@router.get("/get_a_avail_type_gift")
def get_a_avail_type_gift(
    type_gift: str,
    user_id: int,
):
    err, gifts = Gift.get_a_avail_type_gift(type_gift)

    if err :
        return rcode(err)
    if gifts._id:
        err, _ = Gift.update_type_gift(gifts._id)

        if err:
            return rcode(err)
        
        err, _ = User.update_score(user_id=user_id,score=-gifts.price)
    
    return {
        **rcode(1000),
        'gifts': gifts
    }

@router.get("/disticnt_gift")
def get_disticnt_gift():
    pass

@router.get("/delete_gift")
def delete_gift(gift_id):
    err , _ = Gift.delete_gift_code(gift_id=gift_id)

    if err:
        return rcode(err)
    
    return rcode(1000)


