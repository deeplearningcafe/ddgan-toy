import os
import sys
import logging


class Logger:
    """Configures global root logging to both console and disk."""

    @staticmethod
    def setup_logging(save_dir: str, filename: str = "run_log.txt"):
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, filename)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

        logging.info(f"=== Logging Initialized: {log_path} ===")
