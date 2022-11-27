from typing import overload
from multipledispatch import dispatch
from logging import getLogger

logger = getLogger("app")


class ErrorCode(object):
    def __init__(self, int_code, str_code, message):

        self.int_code = int_code
        self.str_code = str_code
        self.message = message

    def get_print(self):
        return {
            "code": self.int_code,
            "str_code": self.str_code,
            "message": self.message,
        }


class ErrorList(object):
    def __init__(self):
        self.int_list = {}
        self.str_list = {}
        err = ErrorCode(613, "IndexOutOfRange", "Index out of range")
        self.str_list["IndexOutOfRange"] = err
        self.int_list[613] = err

    def add_error(self, new_error):
        self.int_list[new_error.int_code] = new_error
        self.str_list[new_error.str_code] = new_error

    @dispatch(int)
    def __call__(self, code):
        return self.int_list[code].get_print()

    @dispatch(int)
    def __call__(self, code):
        return self.int_list[code].get_print()

    @dispatch(str)
    def __call__(self, code):
        if code.isdigit():
            code = int(code)
            if code not in self.int_list:
                logger.error("Index out of range: %s" % code)
                return {
                    **self.str_list["IndexOutOfRange"].get_print(),
                    **{
                        "detail": f"Code {code} is not exists in error list"
                    }
                }
            return self.int_list[code].get_print()
        if code not in self.str_list:
            logger.error("Index out of range: %s" % code)
            return {
                **self.str_list["IndexOutOfRange"].get_print(),
                **{
                    "detail": f"Code {code} is not exists in error list"
                }
            }
        return self.str_list[code].get_print()


rcode = ErrorList()

c404 = ErrorCode(int_code=404, str_code="NotFound", message="Not Found")
rcode.add_error(c404)

c401 = ErrorCode(int_code=401, str_code="Unauthorized", message="Unauthorized")
rcode.add_error(c401)

c604 = ErrorCode(int_code=604, str_code="LoginFailed", message="Login Failed")
rcode.add_error(c604)


c605 = ErrorCode(int_code=605, str_code="LoginSuccess",
                 message="Login Successful")
rcode.add_error(c605)

c606 = ErrorCode(int_code=606, str_code="PermissionDenied",
                 message="Permission Denied")
rcode.add_error(c606)

c607 = ErrorCode(int_code=607, str_code="SignatureExpired",
                 message="Signature Expired")
rcode.add_error(c607)

c608 = ErrorCode(int_code=608, str_code="BadSignature",
                 message="Bad Signature")
rcode.add_error(c608)

c609 = ErrorCode(int_code=609, str_code="WrongInput", message="Wrong Input")
rcode.add_error(c609)

c610 = ErrorCode(int_code=610, str_code="UserNameNotFound",
                 message="UserName/UserID Not Found")
rcode.add_error(c610)

c614 = ErrorCode(int_code=614, str_code="NotFound", message="Not Found")
rcode.add_error(c614)

c618 = ErrorCode(int_code=618, str_code="EncodeTokenFailed",
                 message="Encode Token Failed")
rcode.add_error(c618)

c619 = ErrorCode(int_code=619, str_code="InvalidBearer",
                 message="Invalid Bearer")
rcode.add_error(c619)

c620 = ErrorCode(int_code=620, str_code="UserNameExisted",
                 message="UserName/Email Existed")
rcode.add_error(c620)

c621 = ErrorCode(int_code=621, str_code="MoreThanOneFaceInImage",
                 message="There are more than one face in image")
rcode.add_error(c621)

c712 = ErrorCode(int_code=712, str_code="FaceNotFound",
                 message="Face Not Found")
rcode.add_error(c712)

c714 = ErrorCode(int_code=714, str_code="CanNotReadImageFromBytes",
                 message="Can Not Read Image From Bytes")
rcode.add_error(c714)

c900 = ErrorCode(int_code=900, str_code="SQLExecuteError",
                 message="SQL Execute Error")
rcode.add_error(c900)

c901 = ErrorCode(
    int_code=901,
    str_code="ForeignConstraintFails",
    message="A Foreign Key Constraint Fails",
)
rcode.add_error(c901)

c903 = ErrorCode(
    int_code=903, str_code="SQLDataExists", message="SQL Data Exists"
)
rcode.add_error(c903)


c906 = ErrorCode(
    int_code=906,
    str_code="NotFoundInSQL",
    message="Not Found In SQL",
)
rcode.add_error(c906)

c910 = ErrorCode(
    int_code=910,
    str_code="SQLProgrammingError",
    message="SQL: Programming Got Error",
)
rcode.add_error(c910)

c911 = ErrorCode(
    int_code=911,
    str_code="SQLIDNotFound",
    message="SQL: ID Not Found",
)
rcode.add_error(c911)

c1000 = ErrorCode(int_code=1000, str_code="Done", message="Done")
rcode.add_error(c1000)

c1001 = ErrorCode(
    int_code=1001,
    str_code="ServerGotUndeterminedError",
    message="Server Got Undetermined Error",
)
rcode.add_error(c1001)

c1201 = ErrorCode(int_code=1201, str_code="FileNotFound",
                  message="File Not Found")
rcode.add_error(c1201)

c1202 = ErrorCode(
    int_code=1202, str_code="NotSupportedFile", message="Not Supported File"
)
rcode.add_error(c1202)

c1203 = ErrorCode(int_code=1203, str_code="ReadFileError",
                  message="Read File Error")
rcode.add_error(c1203)

c1204 = ErrorCode(int_code=1204, str_code="WriteFileError",
                  message="Write File Error")
rcode.add_error(c1204)

c1205 = ErrorCode(int_code=1205, str_code="DeleteFileError",
                  message="Delete File Error")
rcode.add_error(c1205)

c1206 = ErrorCode(int_code=1206, str_code="BadZipFile",
                  message="Bad Zip File")
rcode.add_error(c1206)

c1305 = ErrorCode(int_code=1305, str_code="CanNotParseJSONString",
                  message="CanNotParseJSONString")
rcode.add_error(c1305)

c1401 = ErrorCode(
    int_code=1401, str_code="WrongUserPassword", message="Wrong Username or Password"
)
rcode.add_error(c1401)

c1402 = ErrorCode(int_code=1402, str_code="Verify Failed",
                  message="Verify Failed")
rcode.add_error(c1402)

c1501 = ErrorCode(
    int_code=1501, str_code="DataAlreadyExist", message="Data Already Exist"
)
rcode.add_error(c1501)

c1601 = ErrorCode(
    int_code=1601, str_code="CallInternalServiceFailed", message="Call Internal Service Failed"
)
rcode.add_error(c1601)

