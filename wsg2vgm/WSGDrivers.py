import numpy as np
import WSG
import zipfile
import json


def uint16_l(data, offset):
    return int.from_bytes(data[offset:offset + 2], byteorder='little')


def uint16_b(data, offset):
    return int.from_bytes(data[offset:offset + 2], byteorder='big')


class Reader:

    @staticmethod
    def get_prom(game, rom_path):
        """ get the game's ROM data from a rom file containing partial data blocks """
        rom_data = bytearray(2 ** 16)  # 64K
        for rom_file in game['rom_files']:
            offset = int(rom_file['offset'], 0)
            filename = rom_file['filename']
            with zipfile.ZipFile(rom_path + game['rom_filename']) as zipf:
                with zipf.open(filename) as f:
                    data = f.read()
            rom_data[offset:offset + len(data)] = data

        return np.frombuffer(rom_data, dtype=np.uint8)

    @staticmethod
    def get_wavetable(game, rom_path):
        """ sample data from a rom file """
        with zipfile.ZipFile(rom_path + game['rom_filename']) as zipf:
            with zipf.open(game['wavetable_filename']) as f:
                wt = np.frombuffer(f.read(), np.uint8)
                return np.reshape(wt, (8, 32))

    def get_game_info(self, game):
        """ driver addresses """
        self.total_songs = game.get('songs_total', 0)
        self.songs = int(game.get('songs_table', '0'), 0)
        self.notes = int(game.get('notes_table', '0'), 0)
        self.volumes = int(game.get('volenv_table', '0'), 0)
        self.volume_length = game.get('volenv_total', 0)
        self.voice_offset_table = int(game.get('voice_offset_table', '0'), 0)
        self.data_addr = int(game.get('data_address', '0'), 0)
        self.waves = int(game.get('waves_table', '0'), 0)
        self.song_offsets = int(game.get('song_offsets', '0'), 0)
        self.note_tuning = int(game.get('note_tuning', '0'), 0)
        self.decay = int(game.get('decay', '0'), 0)
        self.sustain = int(game.get('sustain', '0'), 0)
        self.attack = int(game.get('attack', '0'), 0)
        self.attack_env = int(game.get('attack_env', '0'), 0)
        self.dur_multiplier = int(game.get('dur_multiplier', '0'), 0)

    def __init__(self, game_name):
        self.game_name = game_name
        self.loop_end = 60 * 60 * 2  # 2 minutes max

    def read(self, song_nr):
        try:
            with open('json/games_info.json') as infile:
                games_info = json.loads(infile.read())
                self.rom_path = games_info.get('rom_path', '')
                game = next((item for item in games_info['games'] if item['game_name'] == self.game_name), None)
                if game:
                    self.get_game_info(game)
                    self.rom = Reader.get_prom(game, self.rom_path)
                    if game.get('wavetable_filename'):
                        self.wavetable = Reader.get_wavetable(game, self.rom_path)

            with open('json/' + self.game_name + '.json') as f:
                data = json.loads(f.read())
                rom_info = data.get('rom_info')
                if rom_info:
                    self.get_game_info(rom_info)
                    self.rom = Reader.get_prom(rom_info, self.rom_path)
                    self.wavetable = Reader.get_wavetable(rom_info, self.rom_path)
                songs = data.get('songs')
                if songs and song_nr < len(songs):
                    self.loop_end = songs[song_nr].get('loop_end', self.loop_end)

        except IOError:
            pass

        if song_nr >= self.total_songs:
            raise Exception('Song nr exceeds the total!')

        if self.game_name == 'ponpoko':
            return self.read_ponpoko(song_nr)
        elif self.game_name in ('superpacm', 'pacnpal'):
            return self.read_superpacm(song_nr)
        elif self.game_name == 'phozon':
            return self.read_phozon(song_nr)
        elif self.game_name in ('grobda', 'liblrabl'):
            return self.read_grobda(song_nr)
        elif self.game_name == 'mappy':
            return self.read_mappy(song_nr)
        elif self.game_name in ('todruaga', 'digdug2', 'motos', 'toypop'):
            return self.read_todruaga(song_nr)
        elif self.game_name in ('skykid', 'drgnbstr', 'metrocrs', 'pacland', 'baraduke'):
            return self.read_skykid(song_nr)
        else:
            raise Exception('Wrong game name!')

    def read_ponpoko(self, song_nr):
        """ Ponpoko is using the original 3OSC WSG driven by Z80. The game features 12 tunes which comprise both
        special effects (tune 1-8) and in-game music (9-12). Ponpoko uses a low level representation where the pitch
        and volume are represented as direct register values. Each change of either pitch of volume is noted as an
        event which stores both values. The only form of compression is in form of event duration allowing for
        representing times up to ~4.2 s (or 254 vblank ticks). Longer durations are obtained by repeated events. In
        particular, individual events consist of 6 (track 0) or 5 bytes (track 1&2) where the first byte indicates
        duration (0x00-0xFE in vblank ticks) or end of track (0xFF), the second byte indicates volume (0x00-0x0F) and
        the remaining bytes store the register frequency value (4 bytes for track 0 and 3 bytes for track 1&2). The
        register value is in its unpacked version so that only low nibbles of each byte are considered. The order is
        little-endian so the least significant byte first.The format also allows specifying a selected waveform for
        each track which is stored separately. This way, the waveform for each track is fixed for the whole duration
        of the song. The waveform data is a set of 3 bytes for each song with a separate location in the memory. """

        event_length = [6, 5, 5]
        tracks = []

        song_addr = uint16_l(self.rom, self.songs + song_nr * 2)
        wave_addr = uint16_l(self.rom, self.waves + song_nr * 2)

        for i in range(3):
            timestamp = 0
            track = []
            volume = -1
            freq = 0
            prev_note = []
            track_addr = uint16_l(self.rom, song_addr + i * 2)

            if i == 0:
                track.append(WSG.Wavetable(timestamp, self.wavetable))
                track.append(WSG.SampleRate(timestamp, 96000))  # 96 kHz
                track.append(WSG.FrameRate(timestamp, 18432000 / 3 / (384 * 264)))  # 60.6... Hz
                track.append(WSG.RegisterSize(timestamp, 20))
            else:
                track.append(WSG.RegisterSize(timestamp, 16))

            # wave number
            track.append(WSG.Wave(timestamp, self.rom[wave_addr + i]))

            while True:
                if self.rom[track_addr] == 0xFF:
                    break

                # volume
                if volume != self.rom[track_addr + 1]:
                    volume = self.rom[track_addr + 1]
                    track.append(WSG.Volume(timestamp, volume))

                #note
                duration = self.rom[track_addr]
                val = 0
                for num in range(2, event_length[i]):
                    val += self.rom[track_addr + num] << ((num - 2)*4)

                track.append(WSG.Note(timestamp, val, duration))
                # if val != 0:
                #     if val != freq:
                #         freq = val
                #         track.append(WSG.Note(timestamp, freq, duration))
                #         prev_note = track[-1]
                #     else:
                #         prev_note.duration += duration
                #         print('!')

                timestamp += duration
                track_addr += event_length[i]

            tracks.append(track)

        return tracks

    def read_superpacm(self, song_nr):
        rom = self.rom

        song_off = rom[self.song_offsets + song_nr * 4]
        nr_tracks = rom[self.song_offsets + song_nr * 4 + 2]

        track_addr = []
        tracks = []

        for i in range(nr_tracks):
            offset = self.songs + (song_off + i) * 2
            track_addr.append(int.from_bytes(rom[offset:offset + 2], byteorder='big'))

        for num, start_addr in enumerate(track_addr):
            track = []
            timestamp = 0
            track_off = song_off + num
            scale_nr = rom[self.note_tuning + track_off]
            offset = self.notes + scale_nr * 2
            note_addr = int.from_bytes(rom[offset:offset + 2], byteorder='big')
            wave_nr = rom[self.waves + track_off] >> 4
            sustain_len = rom[self.sustain + track_off]
            decay_len = rom[self.decay + track_off]
            attack_len = rom[self.attack + track_off]
            offset = self.attack_env + attack_len * 2
            attack_addr = int.from_bytes(rom[offset:offset + 2], byteorder='big')
            attack_len <<= 2
            current_volume = 0
            prev_volume = current_volume
            note_duration = 0
            current_note = 0
            attack_cnt = 0
            sustain_cnt = 0
            decay_cnt = 0

            if num == 0:
                track.append(WSG.Wavetable(timestamp, self.wavetable))
                track.append(WSG.SampleRate(0, 24000))
                track.append(WSG.FrameRate(0, 18432000 / 3 / (384 * 264)))
            track.append(WSG.RegisterSize(0, 20))
            track.append(WSG.Wave(timestamp, wave_nr))

            while rom[start_addr] != 0xFF or note_duration:
                if note_duration == 0:
                    # get a register value from the note lookup
                    offset = note_addr + (rom[start_addr] >> 4) * 4
                    current_note = int.from_bytes(rom[offset:offset + 4], byteorder='big')
                    # apply octave divider
                    current_note >>= (rom[start_addr] & 0xF)
                    current_volume = 0xC
                    note_duration = rom[start_addr + 1]
                    track.append(WSG.Note(timestamp, current_note, note_duration))
                    start_addr += 2
                    attack_cnt = 0
                    decay_cnt = 0
                    sustain_cnt = 0
                else:
                    if attack_cnt < attack_len:
                        attack_cnt += 1
                        current_volume = rom[attack_addr + attack_cnt]
                    elif sustain_cnt < sustain_len:
                        if sustain_cnt == 0:
                            current_volume = 0xC
                        sustain_cnt += 1
                    elif decay_cnt < decay_len:
                        decay_cnt += 1
                        current_volume -= 1

                    if not current_note:
                        current_volume = 0

                    if prev_volume != current_volume:
                        track.append(WSG.Volume(timestamp, current_volume))
                        prev_volume = current_volume

                    note_duration -= 1
                    timestamp += 1

            tracks.append(track)

        return tracks

    def read_phozon(self, song_nr):
        rom = self.rom

        # calculate address limits for all songs
        track_offset = [rom[self.song_offsets + i * 4] for i in range(self.total_songs)]
        track_nr = [rom[self.song_offsets + i * 4 + 2] for i in range(self.total_songs)]

        event_addr = []
        tracks = []
        timestamp_max = float('inf')

        for i in range(track_nr[song_nr]):
            offset = self.songs + (track_offset[song_nr] + i) * 2
            event_addr.append(uint16_b(rom, offset))

        for num, start_addr in enumerate(event_addr):
            # wave and volume
            timestamp = 0
            track = []
            track_volume = 0x0F
            wave_nr = rom[self.waves + track_offset[song_nr] + num] >> 4
            if num == 0:
                track.append(WSG.Wavetable(timestamp, self.wavetable))
                track.append(WSG.SampleRate(0, 24000))
                track.append(WSG.FrameRate(0, 18432000 / 3 / (384 * 264)))

            track.append(WSG.RegisterSize(0, 20))
            track.append(WSG.Wave(timestamp, wave_nr))
            track.append(WSG.Volume(timestamp, track_volume))

            while rom[start_addr] != 0xFF:
                # get a register value from the note lookup
                offset = self.notes + (rom[start_addr] >> 4) * 4
                value = int.from_bytes(rom[offset:offset + 4], byteorder='big')
                # apply octave divider
                value >>= (rom[start_addr] & 0xF)
                note_duration = rom[start_addr + 1]
                track.append(WSG.Note(timestamp, value, note_duration))
                timestamp += note_duration
                start_addr += 2

            timestamp_max = min(timestamp_max, timestamp)
            tracks.append(track)

        # adjust the final length
        # any track finishing first terminates the song so track lengths need to be adjusted
        for track in tracks:
            for event in reversed(track):
                if event.timestamp > timestamp_max:
                    track.pop()
                elif event.__class__.__name__ == 'Note':
                    event.duration = min(event.duration, timestamp_max - event.timestamp + 1)
                    break

        return tracks

    def read_grobda(self, song_nr):
        rom = self.rom

        offset = self.songs + song_nr * 2
        track_addr = uint16_b(rom, offset)

        tracks = []
        event_addr = []
        note_addr = []

        while rom[track_addr] != 0x11:
            event_addr.append(uint16_b(rom, track_addr))
            # set the note lookup for the track [-12 0 +12 cents]
            note_addr.append(uint16_b(rom, self.notes + rom[track_addr + 2] * 2))
            track_addr += 3

        for num, start_addr in enumerate(event_addr):
            track = []
            repeats = 0
            nonrepeats = 0
            nonrepeats_2 = 0
            timestamp = 0
            if num == 0:
                track.append(WSG.Wavetable(timestamp, self.wavetable))
                track.append(WSG.SampleRate(0, 24000))
                track.append(WSG.FrameRate(0, 18432000 / 3 / (384 * 264)))

            track.append(WSG.RegisterSize(0, 20))
            track.append(WSG.Volume(timestamp, 0xF))
            track.append(WSG.Wave(timestamp, rom[start_addr] >> 4))
            track.append(WSG.VolumeCommand(timestamp, rom[start_addr + 1]))
            vol_addr = uint16_b(rom, self.volumes + rom[start_addr + 1] * 2)
            duration_multiplier = rom[self.dur_multiplier + song_nr]
            start_addr += 2
            duration = 0
            prev_volume = 0
            current_volume = 0
            vol_index = 0
            ignore_env = 0
            ignore_jump = 0

            while True:
                if duration == 0:
                    if rom[start_addr] == 0xF0:
                        break
                    elif rom[start_addr] == 0xF1:
                        # wave
                        track.append(WSG.Wave(timestamp, rom[start_addr + 1] >> 4))
                        start_addr += 2
                    elif rom[start_addr] == 0xF2:
                        # volume command
                        track.append(WSG.VolumeCommand(timestamp, rom[start_addr + 1]))
                        vol_addr = uint16_b(rom, self.volumes + rom[start_addr + 1] * 2)
                        ignore_env = 0
                        start_addr += 2
                    elif rom[start_addr] == 0xF3:
                        # conditional jump nr of times
                        repeats += 1
                        if rom[start_addr + 1] <= repeats or ignore_jump:
                            repeats = 0
                            start_addr += 4
                        else:
                            start_addr = uint16_b(rom, start_addr + 2)
                    elif rom[start_addr] == 0xF4:
                        # ignore conditional jump F3
                        ignore_jump = 1
                        start_addr += 2
                    elif rom[start_addr] == 0xF5:
                        # conditional jump after x times
                        nonrepeats += 1
                        if rom[start_addr + 1] == nonrepeats:
                            start_addr = uint16_b(rom, start_addr + 2)
                            nonrepeats = 0
                        else:
                            start_addr += 4
                    elif rom[start_addr] == 0xF6:
                        # conditional jump
                        nonrepeats_2 += 1
                        if rom[start_addr + 1] == nonrepeats_2:
                            start_addr = uint16_b(rom, start_addr + 2)
                            nonrepeats_2 = 0
                        else:
                            start_addr += 4
                    elif rom[start_addr] == 0xF7:
                        # unconditional jump
                        start_addr = uint16_b(rom, start_addr + 1)
                    elif rom[start_addr] >= 0xF0:
                        raise Exception('Unrecognised command %02X' % (rom[start_addr]))
                    else:
                        # get a register value from the note lookup
                        current_note = 0
                        if rom[start_addr] >> 4 != 0xC:
                            offset = note_addr[num] + (rom[start_addr] >> 4) * 3
                            current_note = int.from_bytes(rom[offset:offset + 3], byteorder='big')
                            # apply octave divider
                            current_note >>= (rom[start_addr] & 0xF)
                        duration = int(rom[start_addr + 1]) * duration_multiplier
                        track.append(WSG.Note(timestamp, current_note, duration))
                        if ignore_env == 0:
                            vol_index = vol_addr
                        start_addr += 2
                else:
                    # volume processing for each channel
                    value = rom[vol_index]
                    if value < 0x10:
                        current_volume = value
                        vol_index += 1
                    elif value == 0x12:
                        if current_volume >= duration:
                            current_volume = duration - 1
                    elif value == 0x14:
                        vol_index = vol_addr
                        current_volume = rom[vol_index]
                        vol_index += 1
                    elif value == 0x16:
                        if current_volume > rom[vol_index + 1]:
                            current_volume -= 1
                        else:
                            current_volume = rom[vol_index + 1]
                            vol_index += 2
                    else:
                        if value != 0x10:
                            raise Exception('Unsupported volume command %02X' % value)

                    if prev_volume != current_volume:
                        track.append(WSG.Volume(timestamp, current_volume))
                        prev_volume = current_volume

                    timestamp += 1
                    duration -= 1

            tracks.append(track)

        return tracks

    def read_mappy(self, song_nr):
        rom = self.rom

        offset = self.songs + song_nr * 2
        patt_addr = uint16_b(rom, offset)
        duration_multiplier = rom[self.dur_multiplier + song_nr]

        note_addr = []
        vol_addr = []
        tracks = []
        timestamp = []

        while rom[patt_addr] != 0x11:
            track_addr = uint16_b(rom, patt_addr)
            patt_addr += 2

            track_id = 0
            # patt_timestamp to keep track of pattern lengths
            patt_timestamp = 0
            if len(timestamp):
                patt_timestamp = max(timestamp)
            while rom[track_addr] != 0x11:
                track = []
                current_note = 0
                start_addr = uint16_b(rom, track_addr)
                # initialise the timestamp variable
                if len(tracks) == 0:
                    track.append(WSG.Wavetable(0, self.wavetable))
                    track.append(WSG.SampleRate(0, 24000))
                    track.append(WSG.FrameRate(0, 18432000 / 3 / (384 * 264)))

                if len(tracks) <= track_id:
                    timestamp.append(patt_timestamp)
                    note_addr.append(uint16_b(rom, self.notes + rom[track_addr + 2] * 2))
                    vol_addr.append(uint16_b(rom, self.volumes + rom[start_addr + 1] * 2))
                    track.append(WSG.RegisterSize(0, 20))

                note_addr[track_id] = uint16_b(rom, self.notes + rom[track_addr + 2] * 2)
                vol_addr[track_id] = uint16_b(rom, self.volumes + rom[start_addr + 1] * 2)

                track.append(WSG.Wave(timestamp[track_id], rom[start_addr] >> 4))
                track.append(WSG.VolumeCommand(timestamp[track_id], rom[start_addr + 1]))

                start_addr += 2

                prev_volume = 0
                duration = 0
                fx_counter = 0

                while True:
                    if duration == 0:
                        if rom[start_addr] == 0xF0:
                            # note tuning
                            note_addr[track_id] = uint16_b(rom, self.notes + rom[start_addr + 1] * 2)
                        elif rom[start_addr] == 0xF1:
                            # wave nr
                            track.append(WSG.Wave(timestamp[track_id], rom[start_addr + 1] >> 4))
                        elif rom[start_addr] == 0xF2:
                            # volume command
                            track.append(WSG.VolumeCommand(timestamp[track_id], rom[start_addr + 1]))
                            vol_addr[track_id] = uint16_b(rom, self.volumes + rom[start_addr + 1] * 2)
                        elif rom[start_addr] == 0xF3:
                            # end of track
                            break
                        elif rom[start_addr] >= 0xF0:
                            raise Exception('Unrecognised command %02X' % (rom[start_addr]))
                        else:
                            # get a register value from the note lookup
                            offset = note_addr[track_id] + (rom[start_addr] >> 4) * 4
                            value = int.from_bytes(rom[offset:offset + 4], byteorder='big')
                            # apply octave divider
                            value >>= (rom[start_addr] & 0xF)
                            current_note = value
                            duration = int(rom[start_addr + 1]) * duration_multiplier
                            track.append(WSG.Note(timestamp[track_id], value, duration))
                            volume_index = vol_addr[track_id]
                            fx_counter = 0

                        start_addr += 2
                    else:
                        value = rom[volume_index]
                        if value < 0x10:
                            current_volume = value
                            volume_index += 1
                        elif value == 0x20:
                            volume_index = vol_addr[track_id]
                            continue
                        elif value == 0x30:
                            if current_volume >= duration:
                                current_volume = duration
                        elif value == 0x40:
                            fx_counter += 1
                            if fx_counter > duration:
                                current_volume = 0
                        elif value == 0x50:
                            if current_volume > rom[volume_index + 1]:
                                current_volume -= 1
                            else:
                                current_volume = rom[volume_index + 1]
                                volume_index += 2
                        else:
                            if value != 0x10:
                                raise Exception('Unsupported volume command %02X' % value)

                        if not current_note:
                            current_volume = 0

                        if prev_volume != current_volume:
                            track.append(WSG.Volume(timestamp[track_id], current_volume))
                            prev_volume = current_volume

                        timestamp[track_id] += 1
                        duration -= 1

                if len(tracks) <= track_id:
                    tracks.append(track)
                else:
                    tracks[track_id].extend(track)

                track_addr += 3
                track_id += 1

        return tracks


    def read_todruaga(self, song_nr):
        # supports the following systems and games:
        #  - Namco Super Pacman: todruaga, digdug2, motos
        #  - Namco System 16 Universal: toypop

        rom = self.rom

        timestamp_max = self.loop_end

        # calculate address limits for all songs
        track_addr = [uint16_b(rom, self.songs + i * 2) for i in range(self.total_songs)]

        # # add hidden tracks for todruaga and digdug2
        # if self.game_name == 'todruaga':
        #     track_addr.extend([0xF46F, 0xF4A4])
        # elif self.game_name == 'digdug2':
        #     track_addr.extend([0xE6D3, 0xE7FF, 0xEA6A, 0xEA77, 0xEAC6, 0xEB38])

        track_addr = track_addr[song_nr]
        event_addr = []
        note_addr = []
        tracks = []
        vol_envelopes = []

        # extract volume envelopes
        for num in range(self.volume_length):
            vol_start = uint16_b(rom, self.volumes + num * 2)
            ind = vol_start
            while True:
                if rom[ind] in {0x10, 0x12, 0x13, 0x14}:
                    vol_envelopes.append(rom[vol_start:ind + 1])
                    break
                ind += 1

        # identify track and volume addresses
        # terminated by 0xE0
        while rom[track_addr] != 0xE0:
            # point to the first event
            event_addr.append(uint16_b(rom, track_addr))
            # set the note lookup for the track [-12 0 +12 cents]
            note_addr.append(uint16_b(rom, self.notes + rom[track_addr + 2] * 2))
            track_addr += 3

        # read all tracks
        for num, index in enumerate(event_addr):
            track = []
            timestamp = 0
            repeats = 0
            nonrepeats = 0
            current_value = 0
            duration = 0
            duration_multiplier = 1
            current_volume = 0
            prev_volume = 0
            vol_start = -1
            vol_index = -1
            vol_ignore = 0

            if num == 0:
                track.append(WSG.Wavetable(timestamp, self.wavetable))
                track.append(WSG.SampleRate(0, 24000))
                track.append(WSG.FrameRate(0, 18432000 / 3 / (384 * 264)))

            track.append(WSG.RegisterSize(0, 20))

            while True:
                if duration == 0:
                    if rom[index] == 0xF0:
                        # wave nr
                        track.append(WSG.Wave(timestamp, rom[index + 1] >> 4))
                        index += 2
                    elif rom[index] == 0xF1:
                        # volume envelope
                        vol_env = rom[index + 1]
                        if vol_env > self.volume_length:
                            print('substitue volume from %02X to %02X' % (vol_env, 0x0E))
                            vol_env = 0x0E
                        vol_start = uint16_b(rom, self.volumes + vol_env * 2)
                        vol_index = vol_start
                        track.append(WSG.VolumeCommand(timestamp, vol_env, vol_envelopes[vol_env]))
                        index += 2
                        vol_ignore = 0  # perhaps this should not be set here?
                    elif rom[index] == 0xF2:
                        # duration multiplier
                        duration_multiplier = rom[index + 1]
                        index += 2
                    elif rom[index] == 0xF3:
                        # end of track
                        break
                    elif rom[index] == 0xF4:
                        # conditional skip x times
                        repeats += 1
                        if rom[index + 1] > repeats:
                            index = uint16_b(rom, index + 2)
                        else:
                            repeats = 0
                            index += 4
                    elif rom[index] == 0xF5:
                        # conditional jump after x times
                        nonrepeats += 1
                        if rom[index + 1] == nonrepeats:
                            index = uint16_b(rom, index + 2)
                            nonrepeats = 0
                        else:
                            index += 4
                    elif rom[index] == 0xF6:
                        # unconditional jump
                        index = uint16_b(rom, index + 1)
                    elif rom[index] == 0xF7:
                        #  reset volume envelope vol_ignore = 0
                        vol_ignore = 0  # not sure that this has any effect as the
                        index += 1
                    # duration multiplier
                    elif rom[index] < 0xF0:
                        # get a register value from the note lookup
                        offset = note_addr[num] + (rom[index] >> 4) * 3
                        current_value = int.from_bytes(rom[offset:offset + 3], 'big')
                        # apply octave divider
                        current_value >>= (rom[index] & 0xF)
                        duration = int(rom[index + 1]) * duration_multiplier
                        if current_value:
                            track.append(WSG.Note(timestamp, current_value, duration))
                        if vol_ignore == 0:
                            vol_index = vol_start
                        index += 2
                    else:
                        raise Exception('Unrecognised command %02X' % (rom[index]))
                else:
                    # volume processing for each channel
                    if rom[vol_index] < 0x10:
                        # direct value
                        current_volume = rom[vol_index]
                        vol_index += 1
                    elif rom[vol_index] == 0x10:
                        # keep the last value (sustain)
                        pass
                    elif rom[vol_index] == 0x11:
                        # volume slide down
                        if current_volume > rom[vol_index + 1]:
                            current_volume -= 1
                        else:
                            current_volume = rom[vol_index + 1]
                            vol_index += 2
                    elif rom[vol_index] == 0x12:
                        # volume fade out, duration dependent
                        if current_volume >= duration:
                            current_volume = duration - 1
                    elif rom[vol_index] == 0x13:
                        # reset envelope, loop
                        vol_index = vol_start
                        current_volume = rom[vol_index]
                        vol_index += 1
                    elif rom[vol_index] == 0x14:
                        # ignore envelope resets, used mainly with fx
                        vol_ignore = 1
                        current_volume = rom[vol_index + 1]
                        vol_index += 2
                    else:
                        raise Exception('Unsupported volume command %02X' % rom[vol_index])

                    if not current_value:
                        current_volume = 0

                    if prev_volume != current_volume:
                        track.append(WSG.Volume(timestamp, current_volume))
                        prev_volume = current_volume

                    timestamp += 1
                    duration -= 1
                    if timestamp > timestamp_max:
                        break

            timestamp_max = min(timestamp, timestamp_max)
            tracks.append(track)

        # adjust the final length
        # any track finishing first terminates the song so track lengths need to be adjusted
        for track in tracks:
            for event in reversed(track):
                if event.timestamp > timestamp_max:
                    track.pop()
                elif event.__class__.__name__ == 'Note':
                    event.duration = min(event.duration, timestamp_max - event.timestamp)
                    break

        return tracks

    def read_skykid(self, song_nr):

        rom = self.rom

        wavetable_addr = uint16_b(rom, self.data_addr)
        self.songs = uint16_b(rom, self.data_addr + 4)
        self.volumes = uint16_b(rom, self.data_addr + 6)
        self.dur_multiplier = uint16_b(rom, self.data_addr + 14)

        wavetable = np.zeros((16, 32))
        for n in range(16):
            for v in range(16):
                value = rom[wavetable_addr + n * 16 + v]
                wavetable[n, v * 2] = (value >> 4)
                wavetable[n, v * 2 + 1] = (value & 0xF)

        track_addr = uint16_b(rom, self.songs + song_nr * 2)

        tracks = []
        event_addr = []
        note_addr = []
        fine_tune = []
        note_transpose = []
        current_wave = []
        current_vol = []
        track_control = []
        timestamp_max = 10000
        duration_multiplier = []

        if self.game_name == 'skykid' and song_nr == 2:
            timestamp_max = 384

        # track structure
        # 00-01 track address
        # 02 voice/osc nr
        # 03 pitch modified XY: X fine tune (6cents), Y note transpose
        # 04 wave info + delay track control
        while rom[track_addr] != 0x11:
            # point to the first event
            new_address = uint16_b(rom, track_addr)
            event_addr.append(new_address)
            # osc nr +2
            # pitch +3
            pitch_info = rom[track_addr + 3]
            fine_tune.append(pitch_info >> 4)
            note_transpose.append(pitch_info & 0xF)
            # wave +4
            current_wave.append(rom[track_addr + 4] >> 4)
            track_control.append(rom[track_addr + 4] & 0xF)
            # vol_instr +5
            current_vol.append(rom[track_addr + 5])
            track_addr += 6
            if self.game_name == 'baraduke':
                track_addr += 1

        skiptime = [0] * len(event_addr)
        duration_multiplier.append((0, rom[self.dur_multiplier + song_nr]))

        for num, start_addr in enumerate(event_addr):
            track = []
            timestamp = 0
            nonrepeats = 0
            repeats = 0
            repeats_2 = 0
            note_duration = 0
            current_volume = 0
            prev_volume = -1
            track.append(WSG.VolumeCommand(timestamp, current_vol[num]))
            vol_addr =  uint16_b(rom, self.volumes + current_vol[num] * 2)
            vol_index = vol_addr

            if num == 0:
                track.append(WSG.Wavetable(timestamp, wavetable))
                track.append(WSG.SampleRate(timestamp, 24000))
                track.append(WSG.FrameRate(0, 18432000 / 3 / (384 * 264)))

            track.append(WSG.RegisterSize(timestamp, 20))
            track.append(WSG.Wave(timestamp, current_wave[num]))
            special_mode = 0
            timestamp = skiptime[num]
            cwave = current_wave[num]

            while rom[start_addr] != 0xE0 or note_duration:
                if special_mode and rom[start_addr] == 0:
                    start_addr += 1
                    continue
                if note_duration == 0:
                    if rom[start_addr] > 0xE0:
                        if rom[start_addr] == 0xE1:
                            current_wave[num] = (rom[start_addr + 1] >> 4)
                            cwave = current_wave[num]
                            track.append(WSG.Wave(timestamp, cwave))
                            start_addr += 2
                        elif rom[start_addr] == 0xE3:
                            track.append(WSG.VolumeCommand(timestamp, rom[start_addr + 1]))
                            vol_addr = uint16_b(rom, self.volumes + rom[start_addr + 1] * 2)
                            start_addr += 2
                        elif rom[start_addr] == 0xE4:
                            current_vol[num] += rom[start_addr + 1]
                            current_vol[num] &= 0xF
                            track.append(WSG.VolumeCommand(timestamp, current_vol[num]))
                            vol_addr = uint16_b(rom, self.volumes + current_vol[num] * 2)
                            start_addr += 2
                        elif rom[start_addr] == 0xE5:
                            repeats += 1
                            if rom[start_addr + 1] > repeats:
                                start_addr = uint16_b(rom, start_addr + 2)
                            else:
                                repeats = 0
                                start_addr += 4
                        elif rom[start_addr] == 0xE7:
                            repeats_2 += 1
                            if rom[start_addr + 1] > repeats_2:
                                start_addr = uint16_b(rom, start_addr + 2)
                            else:
                                start_addr += 4
                        elif rom[start_addr] == 0xE8:
                            nonrepeats += 1
                            if rom[start_addr + 1] == nonrepeats:
                                start_addr = uint16_b(rom, start_addr + 2)
                                nonrepeats = 0
                            else:
                                start_addr += 4
                        elif rom[start_addr] == 0xE9:
                            start_addr = uint16_b(rom, start_addr + 1)
                        elif rom[start_addr] == 0xEA:
                            break
                            # noise on
                            special_mode = 1
                            start_addr += 1
                        elif rom[start_addr] == 0xEF:
                            if track_control[num]:
                                track_control[num] = 0
                            else:
                                track_control[num] = rom[start_addr + 1]
                            start_addr += 2
                        elif rom[start_addr] == 0xF0:
                            if track_control[num]:
                                for t in range(track_control[num]):
                                    skiptime[t + num + 1] = timestamp
                            start_addr += 1
                        elif rom[start_addr] == 0xEB:
                            # noise off
                            start_addr += 1
                        # master track
                        # changing global duration multiplier for all tracks
                        elif rom[start_addr] == 0xF1:
                            duration_multiplier.append((timestamp, duration_multiplier[-1][1] + rom[start_addr + 1]))
                            start_addr += 2
                        elif rom[start_addr] == 0xF2:
                            duration_multiplier.append((timestamp, rom[self.dur_multiplier + song_nr]))
                            start_addr += 1
                        elif rom[start_addr] > 0xE0:
                            raise Exception('Unknown command %02X' % rom[start_addr])
                        continue
                    else:
                        if cwave != current_wave[num]:
                            cwave = current_wave[num]
                            track.append(WSG.Wave(timestamp, cwave))
                        value = 0
                        if (rom[start_addr] >> 4) < 0xC:
                            offset = self.notes + ((rom[start_addr] >> 4) + note_transpose[num]) * 3
                            value = int.from_bytes(rom[offset:offset + 3], byteorder='big')
                            value += fine_tune[num] * (value >> 8)
                            # apply octave divider
                            value >>= (rom[start_addr] & 0xF)
                        dmult = 1
                        for dm in duration_multiplier:
                            if dm[0] <= timestamp:
                                dmult = dm[1]
                            else:
                                break
                        note_duration = int(rom[start_addr + 1]) * dmult
                        note_duration &= 0xFF
                        track.append(WSG.Note(timestamp, value, note_duration))
                        index_note = len(track) - 1
                        start_addr += 2
                        vol_index = vol_addr

                # pitch correction
                if rom[vol_index] == 0x1E:
                    rate = int.from_bytes(rom[vol_index + 1], byteorder='big', signed=True)
                    for i in range(abs(rate)):
                        if rate > 0:
                            value += (value >> 8)
                        else:
                            value -= (value >> 8)
                    dt = timestamp - track[index_note].timestamp
                    new_duration = track[index_note].duration - dt
                    track[index_note].duration = dt
                    track.append(WSG.Note(timestamp, value, new_duration))
                    index_note = len(track) - 1
                    vol_index += 2
                    continue
                elif rom[vol_index] == 0x1C:
                    cwave += (rom[vol_index + 1] >> 4)
                    cwave &= 0xF
                    track.append(WSG.Wave(timestamp, cwave))
                    vol_index += 2
                    continue

                if note_duration and value == 0:
                    current_volume = 0
                else:
                    vol_value = rom[vol_index]
                    if vol_value < 0x10:
                        current_volume = vol_value
                        vol_index += 1
                    elif vol_value == 0x12:
                        if current_volume >= note_duration:
                            current_volume = note_duration - 1
                    elif vol_value == 0x14:
                        vol_index = vol_addr
                        current_volume = rom[vol_index]
                        vol_index += 1
                    elif vol_value == 0x16:
                        if current_volume > rom[vol_index + 1]:
                            current_volume -= 1
                        else:
                            current_volume = rom[vol_index + 1]
                            vol_index += 2
                    else:
                        if vol_value != 0x10:
                            raise Exception('Unsupported volume command %02X' % vol_value)

                if prev_volume != current_volume:
                    track.append(WSG.Volume(timestamp, current_volume))
                    prev_volume = current_volume

                timestamp += 1
                note_duration -= 1
                if timestamp > timestamp_max:
                    break

            if timestamp < timestamp_max:
                timestamp_max = timestamp

            tracks.append(track)

        return tracks
