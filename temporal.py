"""Consecutive source-time dwell filtering. Gaps reset candidates."""
class DwellFilter:
    def __init__(self,seconds=3):
        if seconds<0:raise ValueError('Dwell must be nonnegative')
        self.seconds=seconds;self.active={};self.emitted=set();self.last=None
    def update(self,timestamp,candidates):
        if self.last is not None and timestamp<self.last:raise ValueError('Timestamps must increase')
        self.last=timestamp;present=set(candidates);new=[]
        for key in set(self.active)-present:
            del self.active[key];self.emitted.discard(key)
        for key in present:
            self.active.setdefault(key,timestamp)
            if timestamp-self.active[key]>=self.seconds and key not in self.emitted:
                self.emitted.add(key);new.append(key)
        return new
