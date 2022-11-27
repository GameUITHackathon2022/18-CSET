import pymysql
from src.utils.configs import get_db_config

db_config = get_db_config()


def get_connection():
    connection = pymysql.connect(
        host=db_config.mysql.host,
        user=db_config.mysql.user,
        password=db_config.mysql.password,
        database=db_config.mysql.database,
        cursorclass=pymysql.cursors.DictCursor,
    )
    return connection


def exec_query(query, mode=None, n=20):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            exec_result = cursor.execute(query)
            result = None
            if mode == "fetchone":
                result = cursor.fetchone()
            elif mode == "fetchall":
                result = cursor.fetchall()
            elif mode == "fetchmany":
                result = cursor.fetchmany(n)
        connection.commit()

    return exec_result, result


def _process_join_result(raw: dict, map: list):
    """
    map: list[
        {prefix, key}
    ]
    """
    return_result = {}
    for m in map:
        if len(m) == 3:
            prefix, map_key, keys = m
        elif len(m) == 2:
            prefix, map_key = m
            keys = []
        keys.extend([k for k in raw.keys() if prefix in k])
        result = {}
        for k in keys:
            result[k.replace(prefix, "")] = raw[k]
            # del raw[k]

        return_result[map_key] = result

    return return_result


def process_join_result(raw, map: list):
    if isinstance(raw, list) or isinstance(raw, tuple):
        if len(raw) == 0:
            return ()
        for i, data in enumerate(raw):
            result = _process_join_result(data, map=map)
            for m in map:
                raw[i][m[1]] = result[m[1]]
        return raw
    elif isinstance(raw, dict):
        result = _process_join_result(raw, map=map)
        for m in map:
            raw[m[1]] = result[m[1]]
        return raw
