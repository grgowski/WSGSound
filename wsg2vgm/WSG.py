import numpy as np


class Register:

    def __init__(self, size):
        self.size = size
        self.value = np.uint32(0)
        self.volume = 0
        self.accumulator = np.uint32(0)

    def AssignWavetable(self, wavetable):
        self.wavetable = wavetable
        self.wavetable_order = np.uint32(np.floor(np.log2(len(self.wavetable))))

    # generate a set of sample and advance the accumulator
    def Generate(self, sample_length, oversampling=0):
        out = np.zeros(sample_length)
        shift_amount = self.size - self.wavetable_order + oversampling
        max_value = np.uint32((2 ** self.size - 1) * (2 ** oversampling))

#        volume_scale = self.volume / 15.0 / 7.5
        volume_scale = self.volume / 15.0

        if self.value:
            for i in range(sample_length):
                self.accumulator += self.value
                self.accumulator &= max_value
                if self.volume:
                    out[i] = self.wavetable[self.accumulator >> shift_amount]
            out = ((out + 1) / 8 - 1) * volume_scale

        return out


class Event:

    def __init__(self, timestamp):
        self.timestamp = timestamp


class Note(Event):

    def __init__(self, timestamp, value, duration):
        super().__init__(timestamp)
        self.value = value
        self.duration = duration

    def __repr__(self):
        return "<dt %d value %05X duration %d>" % (self.timestamp, int(self.value), self.duration)


class Value(Event):

    def __init__(self, timestamp, value):
        super().__init__(timestamp)
        self.value = value

    def __repr__(self):
        return "<dt %d reg value %05X>" % (self.timestamp, self.value)


class Wave(Event):

    def __init__(self, timestamp, value):
        super().__init__(timestamp)
        self.wave = value

    def __repr__(self):
        return "<dt %d wave %X>" % (self.timestamp, self.wave)


class Volume(Event):

    def __init__(self, timestamp, value):
        super().__init__(timestamp)
        self.volume = value

    def __repr__(self):
        return "<dt %d vol %X>" % (self.timestamp, self.volume)


class VolumeCommand(Event):

    def __init__(self, timestamp, value, envelope=[]):
        super().__init__(timestamp)
        self.volume_command = value
        self.envelope = envelope

    def __repr__(self):
        return "<dt %d vol_cmd %X>" % (self.timestamp, self.volume_command)


class DurationMultiplier(Event):

    def __init__(self, timestamp, value):
        super().__init__(timestamp)
        self.duration_multiplier = value

    def __repr__(self):
        return "<dt %d dur_mult %X>" % (self.timestamp, self.duration_multiplier)


class SampleRate(Event):

    def __init__(self, timestamp, rate):
        super().__init__(timestamp)
        self.rate = rate

    def __repr__(self):
        return "<dt %d sample rate %d>" % (self.timestamp, self.rate)


class FrameRate(Event):

    def __init__(self, timestamp, frame_rate):
        super().__init__(timestamp)
        self.frame_rate = frame_rate

    def __repr__(self):
        return "<dt %d frame rate %f>" % (self.timestamp, self.frame_rate)


class RegisterSize(Event):

    def __init__(self, timestamp, size):
        super().__init__(timestamp)
        self.size = size

    def __repr__(self):
        return "<dt %d register size %d b>" % (self.timestamp, self.size)


class Wavetable(Event):

    def __init__(self, timestamp, wavetable):
        super().__init__(timestamp)
        self.wavetable = wavetable

    def __repr__(self):
        return "<dt %d wavetable>" % (self.timestamp)
