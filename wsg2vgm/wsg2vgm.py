import WSGDrivers
import VGM
import json
import numpy as np
import argparse
import ntpath
import gzip


def timestamp_max(tracks):
    value = 0

    for track in tracks:
        for event in reversed(track):
            if event.__class__.__name__ == 'Note':
                value = max(value, event.timestamp + event.duration)
                break
    return value


def tracks2rows(tracks):

    tmax = timestamp_max(tracks)
    rows = [[[] for i in range(len(tracks))] for j in range(tmax)]

    for num, track in enumerate(tracks):
        for event in track:
            if event.timestamp < tmax:
                rows[event.timestamp][num].append(event)

    return rows


# Initiate the parser
parser = argparse.ArgumentParser('Play Namco 15XX sound files')

parser.add_argument('filename')
parser.add_argument('song_nr', nargs='?', type=int, default=-1)
parser.add_argument("--solo", "-s", nargs='+', type=int)

args = parser.parse_args()

# read data from rom
file_reader = WSGDrivers.Reader(args.filename)
tracks = tracks2rows(file_reader.read(args.song_nr))
# a special case for todruaga song 31 which combines 31 + 26
# with an empty frame in between
if args.filename == 'todruaga' and args.song_nr == 31:
    tracks.append([[], [], [], []])
    tracks_add = tracks2rows(file_reader.read(26))
    for row in tracks_add:
        tracks.append(row)

song_loop = False
loop_offset = 0
loop_offset_bytes = 0

# info data
gd3 = VGM.GD3()

# read config file
try:
    with open('json/' + args.filename + '.json') as f:
        data = json.loads(f.read())

        game_info = data.get('game_info')
        if game_info:
            gd3.author = game_info.get('author', '')
            gd3.game_name = game_info.get('game_title', '')
            gd3.system_name = game_info.get('platform', '')
            gd3.vgm_author = game_info.get('vgm_author', '')
            gd3.notes = game_info.get('notes', '')
            gd3.date = game_info.get('date', '')

        songs = data.get('songs')
        if songs and args.song_nr < len(songs):
            module_name = songs[args.song_nr].get('song_title', '')
            gd3.author = songs[args.song_nr].get('author', gd3.author)
            gd3.track_name = songs[args.song_nr].get('song_title', '')
            song_loop = songs[args.song_nr].get('loop', False)
            loop_offset = songs[args.song_nr].get('loop_offset', 0)

except IOError:
    pass

chip = VGM.C352()

channel_len = len(tracks[0])
noteoff_timestamp = [-1] * channel_len
register_size = [0] * channel_len
track_mute = [False] * channel_len
wave = [-1] * channel_len
song_data = bytes()
data_block = bytes()

frame_dt = 0

if args.solo:
    track_mute = [True] * channel_len
    for solo_track in args.solo:
        if solo_track < len(track_mute):
            track_mute[solo_track] = False

instrument_table = []
instrument_length = [0]

for timestamp, row in enumerate(tracks):

    if song_loop and loop_offset == timestamp:
        loop_offset_bytes = len(song_data)

    song_length = len(song_data)
    for track_nr, track in enumerate(row):
        # key offs for the looped tunes
        if song_loop and loop_offset == timestamp:
            song_data += chip.KeyOff(track_nr)
        if noteoff_timestamp[track_nr] == timestamp:
            song_data += chip.KeyOff(track_nr)

        for event in track:
            event_name = event.__class__.__name__
            if not track_mute[track_nr]:
                if event_name == 'Note' and event.value:
                    note_freq = float(event.value) * sample_rate / (2 ** register_size[track_nr])
                    # handle high pitch notes
                    freq_div = round(note_freq * 2 ** 21 / chip.clock_rate)
                    order = max(freq_div.bit_length() - 16, 0)
                    note_freq = note_freq / (2 ** order)
                    current_wave = (order << 4) | (wave[track_nr] & 0xF)
                    if current_wave not in instrument_table:
                        instrument_table.append(current_wave)
                        instrument_length.append(instrument_length[-1] + 2 ** (5 - order))
                    if current_wave != wave[track_nr]:
                        wave[track_nr] = current_wave
                        instr = instrument_table.index(current_wave)
                        song_data += chip.Wave(track_nr, instrument_length[instr], instrument_length[instr+1]-1)
                    song_data += chip.FreqHz(track_nr, note_freq)
                    song_data += chip.KeyOn(track_nr)
                    noteoff_timestamp[track_nr] = event.timestamp + event.duration
                elif event_name == 'Wave':
                    if event.wave not in instrument_table:
                        instrument_table.append(event.wave)
                        instrument_length.append(instrument_length[-1] + 2 ** 5)
                    if event.wave != wave[track_nr]:
                        wave[track_nr] = event.wave
                        instr = instrument_table.index(event.wave)
                        song_data += chip.Wave(track_nr, instrument_length[instr], instrument_length[instr+1]-1)
                elif event_name == 'Volume':
                    song_data += chip.Volume(track_nr, event.volume << 4)
            if event_name == 'SampleRate':
                sample_rate = event.rate
            elif event_name == 'FrameRate':
                delay_rate = 44100 / event.frame_rate
            elif event_name == 'RegisterSize':
                register_size[track_nr] = event.size
            elif event_name == 'Wavetable':
                wavetable = event.wavetable.copy()
                wavetable <<= 4
                wavetable = wavetable.astype('int8')
                wavetable -= 128

    # clunky, move to the top
    frame_dt += delay_rate
    if song_length != len(song_data):
        song_data += chip.ExecKeys()
    song_data += chip.Delay(round(frame_dt))
    frame_dt -= round(frame_dt)

    #switch off all remaning notes
    if timestamp == len(tracks) - 1:
        for track_nr, note_off in enumerate(noteoff_timestamp):
            if note_off >= timestamp:
                song_data += chip.KeyOff(track_nr)
        song_data += chip.ExecKeys()

song_data += VGM.EndOfSound()

# data block
# extract samples and resample if exceeding the freq range
sample_buffer = np.concatenate([wavetable[inst & 0x0F][::(inst >> 4)+1] for inst in instrument_table])
data_block = chip.DataBlock.FromBuffer(sample_buffer.tobytes())

# vgm header
header = VGM.Header()
header.ChipParams(chip.Params())
header.GD3Offset(len(header.data) + len(data_block) + len(song_data))
header.EOFOffset(len(header.data) + len(data_block) + len(song_data) + len(gd3.get_bytes()))
header.TotalSamples(chip.delay_total)

if song_loop:
    header.Loop(loop_offset_bytes + len(data_block), chip.delay_total)

vgm_data = header.data + data_block + song_data + gd3.get_bytes()

# write the packed version
with gzip.open('{:02d} {:s}.vgz'.format(args.song_nr, gd3.track_name.replace(':', ' -')), 'wb') as f:
    f.write(vgm_data)
