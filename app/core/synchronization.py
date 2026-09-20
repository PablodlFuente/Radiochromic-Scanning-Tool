"""Serialize processor operations without coupling numerical code to Tk."""
from functools import wraps
from threading import RLock


def synchronized(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        if not hasattr(self, "processing_lock"):
            self.processing_lock = RLock()
        with self.processing_lock:
            return method(self, *args, **kwargs)
    return run
