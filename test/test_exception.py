import sys

from airtwinnet.exception.exception import AirTwinNetException


try:
    result = 10 / 0

except Exception as e:
    error = AirTwinNetException(e, sys)

    print(error)