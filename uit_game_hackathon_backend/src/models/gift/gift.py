from logging import getLogger

from src.utils.db_helper import exec_query, process_join_result

logger = getLogger("app")


class Gift:
    def __init__(
            self,
            _id, #auto
            code,    #varchar
            type_gift,    #varchar 
            avail,   #tinyint
            price    #int
    ):
        self._id = _id
        self.code = code
        self.type_gift = type_gift
        self.avail = avail
        self.price = price
    
    def json(self):
        return {
            "_id": self._id,
            "code": self.code,
            "type_gift": self.type_gift,
            "avail": self.avail,
            "price": self.price,
        }

    @staticmethod
    def from_json(_json):
        return Gift (
            _id=_json["_id"],
            code=_json["code"],
            type_gift=_json["type_gift"],
            avail=_json["avail"] ,#if _json["is_done"] is not None else 0
            price=_json["price"]
        )

    # @staticmethod
    # def get(user_name, password):
    #     query = "SELECT * FROM User u {}"
    #     joins = []
    #     map_list = []
    #     joins = " ".join(joins)
    #     query = query.format(joins)
    #     query += ' WHERE u.username="{}" AND u.userpassword="{}" ;'.format(user_name,password)
    #     logger.info(f"Query: {query}")
    #     try:
    #         _, user = exec_query(query, mode="fetchone")
    #         if not user:
    #             return "NotFound", None
    #         if len(map_list):
    #             user = process_join_result(user, map=map_list)
    #         return None, User.from_json(user)
    #     except Exception as err:
    #         logger.error(f"Cannot exec query: {query}")
    #         logger.error(err, exc_info=True)
    #         return "SQLExecuteError", None
    
    @staticmethod
    def get_all_avail( isavail=1  ):
        query = "SELECT * FROM Gift g WHERE g.avail={}".format(isavail)
        # query = "SELECT TOP {} * FROM User".format(top)
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, users = exec_query(query, mode="fetchone")
            if len(map_list):
                users = process_join_result(users, map=map_list)
            return None, list(map(Gift.from_json, users))
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
    # _id, #auto
    #         code,    #varchar
    #         type_gift,    #varchar 
    #         avail,   #tinyint
    #         price    #int
    @staticmethod
    def insert(Gift):
        query = (
            'INSERT INTO Gift(  code, type_gift,avail,price) VALUES( "{}", "{}", "{}","{}");'.format(
                Gift.code,
                Gift.type_gift,
                Gift.avail,
                Gift.price
            )
        )
        logger.info(f"Query: {query}")

        try:
            exec_query(query)
            return None, Gift
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None

    @staticmethod
    def get_a_avail_type_gift( type_gift ):
        query = 'SELECT * FROM Gift g WHERE g.avail={} AND g.type_gift="{}" LIMIT 1'.format(1,type_gift)
        # query = "SELECT TOP {} * FROM User".format(top)
        #get duoc roi thi nen set avail cua no = 0 sau khi gui code ve mail cho user
        map_list = []
    
        logger.info(f"Query: {query}")

        try:
            _, user = exec_query(query, mode="fetchone")
            if not user:
                return "NotFound", None
            if len(map_list):
                user = process_join_result(user, map=map_list)
            return None, Gift.from_json(user)
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None

    @staticmethod
    def update_type_gift( gift_id ):
        query = 'Update Gift g SET g.avail=0 WHERE g._id={}'.format(gift_id)
        # query = "SELECT TOP {} * FROM User".format(top)
        #get duoc roi thi nen set avail cua no = 0 sau khi gui code ve mail cho user
    
        logger.info(f"Query: {query}")

        try:
            exec_query(query, mode="fetchall")
            return None, None
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None

    @staticmethod
    def get_distinct_type():
        query = 'SELECT DISTINCT type_gift FROM Gift'
    
    @staticmethod
    def delete_gift_code(gift_id):
        query = 'DELETE FROM Gift g WHERE g._id={gift_id}'.format(gift_id)

        logger.info(f"Query: {query}")
        try:
            exec_query(query, mode="fetchall")
            return None, None
        except Exception as err:
            logger.error(f"Cannot exec query: {query}")
            logger.error(err, exc_info=True)
            return "SQLExecuteError", None
