"""Small bounded caches for arrays owned by one disposable explorer."""


class ViewCache:
    """Keep only the latest value for each named dependency set.

    Keys must be immutable scalar tuples describing all actual dependencies.
    Cached arrays are private to the explorer: callers must not mutate them.
    A changed key replaces its previous result rather than accumulating slider
    states. ``clear`` releases every retained result when the explorer closes.
    """

    def __init__(self):
        self._entries = {}

    def get(self, name, key, build):
        entry = self._entries.get(name)
        if entry is None or entry[0] != key:
            value = build()
            self._entries[name] = (key, value)
            return value
        return entry[1]

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)
