from logging import getLogger

from src.utils.db_helper import exec_query, process_join_result

logger = getLogger("app")

class Detector:
    def __init__(
        self,
        camera_id,
        user_id,
        confidence,
        email_status,
        image_name
    ):
        self.camera_id = camera_id
        self.user_id = user_id
        self.confidence = confidence
        self.email_status = email_status
        self.image_name = image_name
    
    def json(self):
        return {
            "camera_id": self.camera_id,
            "user_id": self.user_id,
            "confidence": self.confidence,
            "email_status": self.email_status,
            "image_name": self.image_name
        }
    
    @staticmethod
    def from_json(_json):
        return Detector(
            camera_id=_json["camera_id"],
            user_id=_json["user_id"],
            confidence=_json["confidence"],
            email_status=_json["email_status"],
            image_name=_json["image_name"],
        )

    @staticmethod
    def get():
        pass