import logging
import os
import sys

def setup_logger(name: str = "DevOpsGPT", log_level: str = "INFO") -> logging.Logger:
    """
    Configures and returns a unified logger with both console and file handlers.
    
    Args:
        name: Name of the logger.
        log_level: Severity level for logs (DEBUG, INFO, WARNING, ERROR).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    # Set the logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Formatter configuration
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File output handler (dynamic folder creation)
    os.makedirs("data", exist_ok=True)
    log_file_path = os.path.join("data", "devops_gpt.log")
    
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to initialize file logger: {e}")
        
    return logger
