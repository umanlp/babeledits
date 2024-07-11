import os
import logging

from EasyEdit.easyeditor.editors.editor import LOG

def redirect_edit_logs(subdir: str):
    handlers = LOG.handlers
    if len(handlers) == 0:
        raise Exception(f"No log handlers found, this means that redirect_edit_logs was called before the editor initalization. It must be called after, once the logger is set up")
    
    file_handler: logging.FileHandler = LOG.handlers[0]
    new_path = os.path.join("logs", subdir, "run.log")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    file_handler.baseFilename = os.path.join(new_path)
    logging.StreamHandler.__init__(file_handler, file_handler._open())