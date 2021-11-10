import requests


class APIException(Exception):
    def __init__(self, response: requests.models.Response):
        self.response = response

    def raise_for_status(self):
        self.response.raise_for_status()


class MessariException(APIException):
    def __init__(self, response: requests.models.Response):
        super().__init__(response = response)
        jsondata = self.response.json()

        self.elapsed = jsondata['status']['elapsed']
        self.timestamp = jsondata['status']['timestamp']
        self.error_code = jsondata['status']['error_code']
        self.error_message = jsondata['status']['error_message']

    def show_error(self) -> tuple:
        return self.error_code, self.error_message
