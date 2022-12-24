import logging
import logging.config
import time

__all__ = ["configure", "GmtFormatter", "formatter_settings"]


class GmtFormatter(logging.Formatter):
    converter = time.gmtime


formatter_settings = {
    "fmt": "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s:%(module)s: %(message)s",
    "datefmt": "%Y-%m-%dT%H:%M:%S",
}


def configure(level: str = "INFO") -> None:
    logging.addLevelName(logging.WARNING, "WARN")

    config_dict = {
        "version": 1,
        "loggers": {
            "": {"level": level, "handlers": ["error_console"]},
            "sanic.error": {"level": level, "handlers": ["error_console"], "propagate": 0},
            "sanic.root": {"level": level, "handlers": ["error_console"], "propagate": 0},
            "sanic.access": {"level": level, "handlers": ["error_console"], "propagate": 0},
            "kafka": {"level": level, "handlers": ["error_console"]},
        },
        "handlers": {
            "error_console": {
                "class": "logging.StreamHandler",
                "formatter": "generic",
                "stream": "ext://sys.stdout",
            }
        },
        "formatters": {"generic": {"()": GmtFormatter, **formatter_settings}},  # type: ignore
    }

    logging.config.dictConfig(config_dict)
