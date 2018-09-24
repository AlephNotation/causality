import warnings

try:
    # progressbars are an *optional* dependency.
    from tqdm import tqdm as optional_progressbar
except ImportError:
    warnings.warn(
        "Package 'tqdm' not installed => No progressbar available. "
        "To enable progressbars run 'pip3 install tqdm'"
    )
    # If they are not available, we use this mock class instead.

    class optional_progressbar(object):
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __next__(self):
            return next(self.iterable)

        def set_postfix_str(self, postfix):
            pass
