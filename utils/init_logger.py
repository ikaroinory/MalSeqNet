import sys

from loguru import logger


def init_logger(log_name: str = None) -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}'
    )

    if log_name is not None:
        logger.add(
            log_name,
            format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}',
            mode='w'
        )
