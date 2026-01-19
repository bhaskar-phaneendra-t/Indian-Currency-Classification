import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = CustomException.get_detailed_error(
            error_message, error_detail
        )
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error(error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in file [{file_name}] "
            f"at line [{line_number}] "
            f"with message: {error_message}"
        )
