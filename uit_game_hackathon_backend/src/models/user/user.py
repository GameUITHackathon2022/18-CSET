from logging import getLogger

from src.utils.db_helper import exec_query, process_join_result

logger = getLogger("app")


class User:
    def __init__(
            self,
            _id,
            name,
            email,
            password,
            score = 0
    ):
        self._id = _id
        self.score = score
        self.name = name
        self.email = email
        self.password = password
    
    def json(self):
        return {
            "_id": self._id,
            "name": self.name,
            "email": self.email,
            "password": self.password,
            "score": self.score
        }

    @staticmethod
    def from_json(_json):
        return User(
            _id=_json["_id"],
            name=_json["username"],
            email=_json["email"],
            password=_json["user_password"],
            score=_json["score"] if _json["score"] is not None else 0
        )

    @staticmethod
    def get(user_name, password):
        query = "SELECT * FROM User u {}"
        joins = []
        map_list = []
        joins = " ".join(joins)
        query = query.format(joins)
        query += ' WHERE u.username="{}" AND u.user_password="{}" ;'.format(user_name,password)
        logger.info(f"Query: {query}")
        try:
            _, user = exec_query(query, mode="fetchone")
            if not user:
                return "NotFound", None
            if len(map_list):
                user = process_join_result(user, map=map_list)
            return None, User.from_json(user)
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def get_all(
        top = 20
    ):
        query = "SELECT * FROM User u ORDER BY u.score DESC  LIMIT {}".format(top)
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, users = exec_query(query, mode="fetchall")
            if len(map_list):
                users = process_join_result(users, map=map_list)
            return None, list(map(User.from_json, users))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def insert(user):
        query = (
            'INSERT INTO User(username, email, user_password, score) VALUES("{}", "{}", "{}","{}");'.format(
                user.name,
                user.email,
                user.password,
                user.score
            )
        )
        logger.info(f"Query: {query}")

        try:
            exec_query(query)
            return None, user
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    
    @staticmethod
    def update_score(user_id, score):
        query = "UPDATE User u {}"
        joins = []
        map_list = []
        joins = " ".join(joins)
        query = query.format(joins)
        query += 'SET u.score = u.score + {}'.format(score)
        query += ' WHERE  u._id="{}" ;'.format(user_id)
        logger.info(f"Query: {query}")
        try:
            exec_query(query, mode="fetchone")


            return None, None
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None