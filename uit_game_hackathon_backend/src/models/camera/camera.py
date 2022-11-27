from logging import getLogger

from src.utils.db_helper import exec_query, process_join_result

logger = getLogger("app")

class Camera:
    def __init__(
        self,
        name,
        user_name,
        password,
        protocol,
    ):
        self.name = name
        self.user_name = user_name
        self.password = password
        self.protocol = protocol

    def json(self):
        return {
            "name" : self.name,
            "user_name": self.user_name,
            "password": self.password,
            "protocol": self.protocol,
        }
    
    @staticmethod
    def from_json(_json):
        return Camera(
            name=_json["name"],
            user_name=_json["user_name"],
            password=_json["password"],
            protocol=_json["protocol"],
            is_hidden=_json["is_hidden"],
            user_id=_json["user_id"],
        )

    @staticmethod
    def get(user_id,name):
        query = "SELECT * FROM Camera c {}"
        joins = []
        map_list = []
        joins = " ".join(joins)
        query = query.format(joins)
        query += ' WHERE c.userid="{}" AND c.cameraname="{}" ;'.format(user_id,name)
        logger.info(f"Query: {query}")
        try:
            _, user = exec_query(query, mode="fetchone")
            if not user:
                return "NotFound", None
            if len(map_list):
                user = process_join_result(user, map=map_list)
            return None, Camera.from_json(user)
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def get_all():
        query = "SELECT * FROM Camera c ORDER BY c.created_at DESC"
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, users = exec_query(query, mode="fetchall")
            if len(map_list):
                users = process_join_result(users, map=map_list)
            return None, list(map(Camera.from_json, users))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
        
    @staticmethod
    def insert(camera):
        query = (
            'INSERT INTO Camera(cameraname, ip, username, protocol, camera_password) VALUES("{}", "{}", "{}", "{}","{}");'.format(
                camera.name,
                camera.ip,
                camera.user_name,
                camera.protocol,
                camera.password,
            )
        )
        logger.info(f"Query: {query}")
# INSERT INTO Camera(cameraname, ip, username, camera_password, protocol,location_desc) VALUES(
        # "Fake_Camera_1","0.0.0.0","admin","admin","rtsp","Fake_Location_1"
        try:
            exec_query(query)
            return None, camera
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def get_user_cameras(user_id):
        query = "SELECT * FROM Camera c"
        query += ' WHERE c.user_id="{}";'.format(user_id)
        map_list = []
        logger.info(f"Query: {query}")

        try:
            _, cameras = exec_query(query, mode="fetchall")
            if len(map_list):
                cameras = process_join_result(cameras, map=map_list)
            return None, list(map(Camera.from_json, cameras))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
