import logging

console_logger = logging.getLogger("lightning.pytorch.core")
handler = logging.FileHandler("core.log")
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

console_logger.addHandler(handler)


def exception_console_logging(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            console_logger.exception(e)
            raise e

    return wrapper


def start_end_end_console_logging(message):
    def wrapper(func):
        def _wrapper(*args, **kwargs):
            console_logger.info(f"start {message}")
            func(*args, **kwargs)
            console_logger.info(f"end {message}")

        return _wrapper

    return wrapper
