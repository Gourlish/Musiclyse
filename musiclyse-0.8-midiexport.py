#!/usr/bin/env python3
"""Export one six-stem multitrack MIDI file from Musiclyse 0.7 (and later) saved JSON. Standard library only."""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

# General MIDI program numbers (0-based). Audition sounds only.
PROGRAMS = {
    'vocals': 53,   # Choir Aahs
    'bass': 33,     # Fingered Bass
    'guitar': 27,   # Electric Guitar (clean)
    'piano': 0,     # Acoustic Grand Piano
    'other': 89,    # Pad 2 (warm)
}

# General MIDI drum map (MIDI note numbers)
DRUMS = {
    'kick': 36, 'bass drum': 36, 'bd': 36,
    'snare': 38, 'sn': 38, 'side stick': 37, 'rim': 37,
    'hi-hat': 42, 'hihat': 42, 'hi hat': 42, 'closed hi-hat': 42, 'chh': 42,
    'open hi-hat': 46, 'ohh': 46,
    'crash': 49, 'cymbal': 49, 'ride': 51,
    'tom': 45, 'floor tom': 41,
    'clap': 39, 'cowbell': 56,
}

STEMS = ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other')

# 0.7.1-pre6; each False gives the previous behaviour.
# A note that starts while an earlier note of the same pitch is still sounding on the same
# track ends the earlier note there, so neither is cut short by the other's note-off.
TRIM_SAME_PITCH_OVERLAPS = True
# The file's tempo is the song's tempo from the save (when it has one), so bars and beats line
# up in a MIDI editor. Playback timing is the same either way.
TEMPO_FROM_SAVE = True
DEFAULT_TEMPO_BPM = 120

def write_var_len(value):
    """Write a MIDI variable-length quantity."""
    buf = bytearray()
    buf.append(value & 0x7F)
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.reverse()
    return bytes(buf)

def _trimmed_notes(events, sec_to_ticks):
    """[(onset_tick, end_tick, note, vel)] with no two notes of one pitch overlapping: a note
    ends where the next note of its pitch starts. Two notes of one pitch starting on the same
    tick become one (the longer, at the louder velocity). Every note lasts at least one tick."""
    by_note = {}
    for onset_sec, dur_sec, note, vel in events:
        on = sec_to_ticks(onset_sec)
        end = on + max(1, sec_to_ticks(dur_sec))
        by_note.setdefault(note, []).append([on, end, vel])
    out = []
    for note, items in by_note.items():
        items.sort()
        merged = []
        for on, end, vel in items:
            if merged and merged[-1][0] == on:
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2] = max(merged[-1][2], vel)
            else:
                merged.append([on, end, vel])
        for k, (on, end, vel) in enumerate(merged):
            if k + 1 < len(merged) and end > merged[k + 1][0]:
                end = merged[k + 1][0]
            out.append((on, max(end, on + 1), note, vel))
    return out


def song_tempo(data):
    """The save's tempo when it is a usable number, else DEFAULT_TEMPO_BPM."""
    try:
        bpm = float(data.get('bpm') or 0)
    except (TypeError, ValueError):
        return DEFAULT_TEMPO_BPM
    return bpm if 20 <= bpm <= 300 else DEFAULT_TEMPO_BPM


def build_midi_file(tracks_data, ticks_per_quarter=960, tempo_bpm=120):
    """
    tracks_data: list of (name, channel, program, events)
    events: list of (time_seconds, duration_seconds, note, velocity)
    """
    mpqn = int(60_000_000 / tempo_bpm)
    
    def sec_to_ticks(sec):
        if TEMPO_FROM_SAVE:
            # From the tempo actually written to the file, so note times stay exact.
            return int(round(sec * 1_000_000 / mpqn * ticks_per_quarter))
        return int(round(sec * (tempo_bpm / 60.0) * ticks_per_quarter))

    chunks = []
    for i, (name, channel, program, events) in enumerate(tracks_data):
        is_drum = (channel == 9)
        
        midi_events = []
        if TRIM_SAME_PITCH_OVERLAPS:
            for onset_tick, end_tick, note, vel in _trimmed_notes(events, sec_to_ticks):
                midi_events.append((onset_tick, 0x90 | channel, note, vel))
                midi_events.append((end_tick, 0x80 | channel, note, 0))
        else:
            for onset_sec, dur_sec, note, vel in events:
                onset_tick = sec_to_ticks(onset_sec)
                dur_tick = sec_to_ticks(dur_sec)
                if dur_tick < 1:
                    dur_tick = 1
                midi_events.append((onset_tick, 0x90 | channel, note, vel))
                midi_events.append((onset_tick + dur_tick, 0x80 | channel, note, 0))
            
        # Sort by tick, then status (note off 0x80 < note on 0x90)
        midi_events.sort(key=lambda e: (e[0], e[1]))
        
        track_data = bytearray()
        
        # Tempo map on first track
        if i == 0:
            track_data.extend(write_var_len(0))
            track_data.extend(b'\xFF\x51\x03')
            track_data.extend(struct.pack('>I', mpqn)[1:]) # 3 bytes
            
        # Program change (not for drums)
        if not is_drum and program is not None:
            track_data.extend(write_var_len(0))
            track_data.extend(bytes([0xC0 | channel, program]))
            
        last_tick = 0
        for tick, status, note, vel in midi_events:
            delta = tick - last_tick
            if delta < 0:
                delta = 0
            track_data.extend(write_var_len(delta))
            track_data.extend(bytes([status, note & 0x7F, vel & 0x7F]))
            last_tick = tick
            
        # End of track
        track_data.extend(write_var_len(0))
        track_data.extend(b'\xFF\x2F\x00')
        
        chunks.append(b'MTrk' + struct.pack('>I', len(track_data)) + track_data)
        
    ntrks = len(tracks_data)
    header = b'MThd' + struct.pack('>IHHH', 6, 1, ntrks, ticks_per_quarter)
    
    return header + b''.join(chunks)

def parse_stem_midi(data, fixed_velocity=None):
    """
    Parses Musiclyse 0.7 stem_midi format.
    Returns dict of stem -> list of (onset_sec, dur_sec, note, vel)
    """
    stem_midi = data.get('stem_midi', {})
    if not isinstance(stem_midi, dict):
        return {}
        
    result = {}
    for stem, rows in stem_midi.items():
        if not isinstance(rows, list):
            continue
        events = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                t = float(row.get('t', 0))
                d = float(row.get('d', 0))
                v_raw = float(row.get('v', 1.0))
                
                # Scale velocity to 1-127, or use fixed
                if fixed_velocity is not None:
                    vel = fixed_velocity
                else:
                    vel = max(1, min(127, int(round(v_raw * 127))))
                    
                # Handle drums (key 'k' indicates drum type)
                k = row.get('k')
                if k:
                    note = DRUMS.get(str(k).lower().strip(), 38) # default snare
                    events.append((t, d, note, vel))
                else:
                    # Pitched note
                    m = row.get('m')
                    if m is not None:
                        note = int(float(m))
                        if 0 <= note <= 127:
                            events.append((t, d, note, vel))
                    else:
                        # Fallback for Hz frequency
                        hz = row.get('hz')
                        if hz is not None:
                            hz = float(hz)
                            if hz > 0:
                                note = int(round(69 + 12 * math.log2(hz / 440.0)))
                                if 0 <= note <= 127:
                                    events.append((t, d, note, vel))
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        if events:
            result[stem] = events
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export Musiclyse 0.7 saved-song .json files to multitrack MIDI (.mid)."
    )
    parser.add_argument('inputs', nargs='+', help='Musiclyse saved .json files or folders')
    parser.add_argument('-r', '--recursive', action='store_true', help='search folders recursively')
    parser.add_argument('-o', '--out-dir', help='output folder (default: same as input)')
    parser.add_argument('--overwrite', action='store_true', help='replace existing MIDI exports')
    parser.add_argument('--velocity', type=int, choices=range(1, 128), metavar='1-127', 
                        help='fixed velocity for auditioning quiet notes; default: scale saved amplitude')
    
    args = parser.parse_args(argv)
    
    files = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            pattern = '**/*.json' if args.recursive else '*.json'
            files.extend(sorted(p.glob(pattern)))
        else:
            files.append(p)
            
    files = list(dict.fromkeys(p.resolve() for p in files))
    if not files:
        parser.error('no JSON files found')
        
    failed = 0
    for source in files:
        try:
            print(f"Processing {source.name}...")
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, dict) or 'stem_midi' not in data:
                print(f"  Skip: not a Musiclyse 0.7 save (missing stem_midi)")
                failed += 1
                continue
                
            stem_events = parse_stem_midi(data, fixed_velocity=args.velocity)
            if not stem_events:
                print(f"  Skip: no MIDI events found")
                failed += 1
                continue
                
            # Build tracks
            tracks_data = []
            channel_map = {
                'vocals': 0, 'bass': 1, 'guitar': 2, 'piano': 3, 'other': 4
            }
            
            # Add pitched stems first
            for stem in STEMS:
                if stem == 'drums':
                    continue
                if stem in stem_events:
                    ch = channel_map.get(stem, 0)
                    prog = PROGRAMS.get(stem)
                    tracks_data.append((stem, ch, prog, stem_events[stem]))
                    
            # Add drums on channel 10 (index 9)
            if 'drums' in stem_events:
                tracks_data.append(('drums', 9, None, stem_events['drums']))
                
            if not tracks_data:
                print(f"  Skip: no valid stems")
                failed += 1
                continue
                
            midi_bytes = build_midi_file(tracks_data, tempo_bpm=song_tempo(data) if TEMPO_FROM_SAVE
                                         else DEFAULT_TEMPO_BPM)
            
            out_name = source.stem + '.mid'
            if args.out_dir:
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / out_name
            else:
                out_path = source.parent / out_name
                
            if out_path.exists() and not args.overwrite:
                print(f"  Skip: {out_path.name} already exists (use --overwrite)")
                continue
                
            with open(out_path, 'wb') as f:
                f.write(midi_bytes)
            print(f"  Saved: {out_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1
            
    if failed:
        print(f"\nFinished with {failed} error(s).")
        return 1
    print("\nDone.")
    return 0

if __name__ == '__main__':
    sys.exit(main())