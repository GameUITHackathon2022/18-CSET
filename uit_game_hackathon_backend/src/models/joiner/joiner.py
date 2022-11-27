from logging import getLogger
from models.mission.mission import Mission

from src.utils.db_helper import exec_query, process_join_result

logger = getLogger("app")

class Joiner:
    def __init__(
        self,
        userid,
        mission_id
    ):
        self.userid = userid
        self.mission_id = mission_id
    
    def json(self):
        return {
            "userid": self.userid,
            "mission_id": self.mission_id,
        }

    @staticmethod
    def from_json(_json):
        return Joiner(
            userid=_json["userid"],
            mission_id=_json["mission_id"],
        )

    @staticmethod
    def get_all():
        query = "SELECT * FROM Joiner j"
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, joiners = exec_query(query, mode="fetchall")
            if len(map_list):
                joiners = process_join_result(joiners, map=map_list)
            return None, list(map(Joiner.from_json, joiners))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None

    @staticmethod
    def get(user_id, mission_id):
        query = "SELECT * FROM Joiner j {}"
        joins = []
        map_list = []
        joins = " ".join(joins)
        query = query.format(joins)
        query += ' WHERE j.userid="{}" AND u.mission_id="{}" ;'.format(user_id,mission_id)
        logger.info(f"Query: {query}")
        try:
            _, user = exec_query(query, mode="fetchone")
            if not user:
                return "NotFound", None
            if len(map_list):
                user = process_join_result(user, map=map_list)
            return None, Joiner.from_json(user)
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def get_all_by_user_id(user_id):
        query = 'SELECT * FROM Joiner j WHERE j.userid={}'.format(user_id)
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, joiners = exec_query(query, mode="fetchall")
            if len(map_list):
                joiners = process_join_result(joiners, map=map_list)
            return None, list(map(Joiner.from_json, joiners))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def get_all_other_by_user_id(user_id):
        query = 'SELECT * FROM Joiner j WHERE j.userid!={}'.format(user_id)
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, joiners = exec_query(query, mode="fetchall")
            if len(map_list):
                joiners = process_join_result(joiners, map=map_list)
            return None, list(map(Joiner.from_json, joiners))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
        
    @staticmethod
    def get_all_other_by_mission_id(user_id):
        # query = 'SELECT * FROM Joiner j WHERE j.mission_id!={}'.format(user_id)
        query = 'SELECT * FROM Mission m WHERE m._id NOT IN (SELECT Joiner.mission_id FROM Joiner WHERE Joiner.userid={});'.format(user_id)
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, joiners = exec_query(query, mode="fetchall")
            if len(map_list):
                joiners = process_join_result(joiners, map=map_list)
            return None, list(map(Mission.from_json, joiners))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def insert(Joiner):
        query = (
            'INSERT INTO Joiner(userid, mission_id) VALUES({}, {});'.format(
                Joiner.userid,
                Joiner.mission_id,
            )
        )
        logger.info(f"Query: {query}")

        try:
            exec_query(query)
            return None, Joiner
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    