import struct


def EndOfSound():
    return struct.pack('B', 0x66)


class Header:

    def __init__(self):
        self.data = bytearray([0] * 0x100)
        self.data[0x00:0x00 + 4] = bytes('Vgm ', 'utf-8')  # vgm id
        self.data[0x08:0x08 + 4] = struct.pack('<I', 0x171)  # version
        self.data[0x34:0x34 + 4] = struct.pack('<I', len(self.data) - 0x34)  # header size

    def GD3Offset(self, offset):
        self.data[0x14:0x14 + 4] = struct.pack('<I', offset - 0x14)

    def EOFOffset(self, offset):
        self.data[0x04:0x04 + 4] = struct.pack('<I', offset - 0x04)

    def TotalSamples(self, samples):
        self.data[0x18:0x18 + 4] = struct.pack('<I', samples)

    def Loop(self, offset, samples):
        self.data[0x1C:0x1C + 4] = struct.pack('<I', offset + len(self.data) - 0x1C)
        self.data[0x20:0x20 + 4] = struct.pack('<I', samples)

    def ChipParams(self, params):
        for p in params:
            offset = p[0]
            param_value = p[1]
            self.data[offset:offset + len(param_value)] = param_value


class GD3:
    # "Track name (in English characters)\0"
    # "Track name (in Japanese characters)\0"
    # "Game name (in English characters)\0"
    # "Game name (in Japanese characters)\0"
    # "System name (in English characters)\0"
    # "System name (in Japanese characters)\0"
    # "Name of Original Track Author (in English characters)\0"
    # "Name of Original Track Author (in Japanese characters)\0"
    # "Date of game's release written in the form yyyy/mm/dd, or just yyyy/mm or yyyy if month and day is not known\0"
    # "Name of person who converted it to a VGM file.\0"
    # "Notes\0"

    def __init__(self):
        self.track_name = ''
        self.game_name = ''
        self.system_name = ''
        self.author = ''
        self.date = ''
        self.vgm_author = ''
        self.notes = ''

    def get_bytes(self):
        out = bytearray(bytes('Gd3 ', 'utf-8'))
        out += struct.pack('<I', 0x0100)  # version
        out += bytes([0] * 4)
        out += bytes(self.track_name, 'utf-16-le') + bytes([0, 0, 0, 0])
        out += bytes(self.game_name, 'utf-16-le') + bytes([0, 0, 0, 0])
        out += bytes(self.system_name, 'utf-16-le') + bytes([0, 0, 0, 0])
        out += bytes(self.author, 'utf-16-le') + bytes([0, 0, 0, 0])
        out += bytes(self.date, 'utf-16-le') + bytes([0, 0])
        out += bytes(self.vgm_author, 'utf-16-le') + bytes([0, 0])
        out += bytes(self.notes, 'utf-16-le') + bytes([0, 0])
        out[8:8 + 4] = struct.pack('<I', len(out) - 12)  # data length
        return out

class Chip:
    def __init__(self):
        self.delay_total = 0

    def Delay(self, delay):
        self.delay_total += delay
        out = bytes()
        while delay > 0xFFFF:
            out += struct.pack('<BH', 0x61, 0xFFFF)
            delay -= 0xFFFF
        out += struct.pack('<BH', 0x61, delay)
        return out

    def ExecKeys(self):
        return bytes()


class C352(Chip):
    FLG_KEYON   = 0x4000   # Keyon
    FLG_KEYOFF  = 0x2000   # Keyoff
    FLG_FILTER  = 0x0004   # don't apply filter
    FLG_LOOP    = 0x0002   # loop forward

    def __init__(self, clock_freq=24576000, clock_div=288):
        super().__init__()
        self.clock_freq = clock_freq
        self.clock_div = clock_div
        self.clock_rate = self.clock_freq / self.clock_div
        self.delay_total = 0

    def Params(self):
        return [[0xD6, struct.pack('<B', self.clock_div >> 2)],
                [0xDC, struct.pack('<I', self.clock_freq)]]

    class DataBlock:

        @staticmethod
        def FromBuffer(buffer):
            out = struct.pack('<BBB', 0x67, 0x66, 0x92)
            out += struct.pack('<III', len(buffer) + 8, len(buffer), 0)
            out += buffer
            return out

        @staticmethod
        def FromSamples(samples):
            sample_data = bytes()
            for s in samples:
                sample_data += s
            return C352.DataBlock.FromBuffer(sample_data)

    @staticmethod
    def Write(address, value):
        return struct.pack('>BHH', 0xE1, address, value)

    def Voice(self, voice_nr, offset, value):
        return self.Write((voice_nr << 4) | offset, value)

    def ExecKeys(self):
        return self.Write(0x0202, 0x0020)

    def KeyOn(self, voice):
        return self.Voice(voice, 3, C352.FLG_KEYON | C352.FLG_FILTER | C352.FLG_LOOP)

    def KeyOff(self, voice):
        return self.Voice(voice, 3, C352.FLG_KEYOFF | C352.FLG_FILTER | C352.FLG_LOOP)

    def Volume(self, voice, volume_left, volume_right=-1):
        if volume_right == -1:
            volume_right = volume_left
        return self.Voice(voice, 0, (volume_left << 8) | volume_right)

    def FreqHz(self, voice, note_freq):
        freq_div = round(note_freq * 2 ** 21 / self.clock_rate)  # 5 from sample length, 16 bit counter
        if freq_div > 0xFFFF:
            print('Freq 0x%06X exceeds the max value' % freq_div)
            print('Required bit length ', freq_div.bit_length() - 16)
            freq_div = 0xFFFF
        return self.FreqDiv(voice, freq_div)

    def FreqDiv(self, voice, freq_div):
        return self.Voice(voice, 2, freq_div)

    def Wave(self, voice, wave_start, wave_end, wave_loop=-1, wave_bank=0):
        if wave_loop == -1:
            wave_loop = wave_start
        out = self.Voice(voice, 4, wave_bank)
        out += self.Voice(voice, 5, wave_start)
        out += self.Voice(voice, 6, wave_end)
        out += self.Voice(voice, 7, wave_loop)
        return out


class C140(Chip):
    def __init__(self, clock_rate=round((49152000 / 384) / 6)):
        super().__init__()
        self.clock_rate = clock_rate

    def Params(self):
        return [[0x96, struct.pack('<B', 0)],
                [0xA8, struct.pack('<I', self.clock_rate)]]

    class DataBlock:

        @staticmethod
        def FromBuffer(buffer):
            out = struct.pack('<BBB', 0x67, 0x66, 0x8D)
            out += struct.pack('<III', len(buffer) + 8, len(buffer), 0)
            out += buffer
            return out

        @staticmethod
        def FromSamples(samples):
            sample_data = bytes()
            for s in samples:
                sample_data += s
            return C140.DataBlock.FromBuffer(sample_data)

    @staticmethod
    def Write(address, value):
        return struct.pack('>BHB', 0xD4, address, value)

    @staticmethod
    def Voice(voice_nr, offset, value):
        return C140.Write((voice_nr << 4) | offset, value)

    @staticmethod
    def Volume(voice, volume_left, volume_right=-1, gain=1/4):
        if volume_right == -1:
            volume_right = volume_left
        volume_left = int(round(volume_left * gain))
        volume_right = int(round(volume_right * gain))
        return C140.Voice(voice, 0, volume_right) + C140.Voice(voice, 1, volume_left)

    @staticmethod
    def FreqDiv(voice, freq_div):
        return C140.Voice(voice, 2, freq_div >> 8) + C140.Voice(voice, 3, freq_div & 0xFF)

    def FreqHz(self, voice, note_freq):
        freq_div = round(note_freq * 2 ** 20 / self.clock_rate)  # 5 from sample length, 16 bit counter
        if freq_div > 0xFFFF:
            print('Freq 0x%06X exceeds the max value' % freq_div)
            freq_div = 0xFFFF
        return self.FreqDiv(voice, freq_div)

    @staticmethod
    def Wave(voice, wave_start, wave_end, wave_loop=-1, wave_bank=0):
        if wave_loop == -1:
            wave_loop = wave_start
        out = C140.Voice(voice, 4, wave_bank)
        out += C140.Voice(voice, 6, wave_start >> 8)
        out += C140.Voice(voice, 7, wave_start & 0xFF)
        out += C140.Voice(voice, 8, wave_end >> 8)
        out += C140.Voice(voice, 9, wave_end & 0xFF)
        out += C140.Voice(voice, 10, wave_loop >> 8)
        out += C140.Voice(voice, 11, wave_loop & 0xFF)
        return out

    @staticmethod
    def KeyOn(voice):
        return C140.Voice(voice, 5, 0xD0)

    @staticmethod
    def KeyOff(voice):
        return C140.Voice(voice, 5, 0x00)
