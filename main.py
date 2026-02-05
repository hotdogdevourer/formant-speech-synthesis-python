#!/usr/bin/env python3
import numpy as np
import scipy.signal as sig
import wave
import re
import sys
import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union

smp = 48000
VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

BYTE_TO_PHONEME = {
    0x00: 'SIL', 0x01: 'AH', 0x02: 'AE', 0x03: 'AA', 0x04: 'AO', 0x05: 'EH', 0x06: 'EY',
    0x07: 'IH', 0x08: 'IY', 0x09: 'OW', 0x0A: 'UH', 0x0B: 'UW', 0x0C: 'ER', 0x0D: 'B',
    0x0E: 'D', 0x0F: 'G', 0x10: 'P', 0x11: 'T', 0x12: 'K', 0x13: 'M', 0x14: 'N', 0x15: 'NG',
    0x16: 'L', 0x17: 'R', 0x18: 'F', 0x19: 'S', 0x1A: 'SH', 0x1B: 'TH', 0x1C: 'DH', 0x1D: 'V',
    0x1E: 'Z', 0x1F: 'ZH', 0x20: 'W', 0x21: 'Y', 0x22: 'HH', 0x23: 'CH', 0x24: 'JH',
}
PHONEME_TO_BYTE = {v: k for k, v in BYTE_TO_PHONEME.items()}
VOWELS = {'AH','AE','AA','AO','EH','EY','IH','IY','OW','UH','UW','ER'}
STOPS = {'P','T','K','B','D','G','CH'}
FRICATIVES_UNVOICED = {'F','S','SH','TH','HH'}
FRICATIVES_VOICED = {'V','Z','ZH','DH'}

class Voice:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.phonemes: Dict[str, Dict] = {}
    
    def get_phoneme_data(self, phoneme: str) -> Dict:
        if phoneme.endswith('_FINAL'):
            base_ph = phoneme.replace('_FINAL', '')
            data = self.phonemes.get(base_ph, self.phonemes.get('SIL', {})).copy()
            if base_ph in VOWELS:
                data['length'] = min(data.get('length', 0.14) * 1.4, 0.35)
            return data
        return self.phonemes.get(phoneme, self.phonemes.get('SIL', {}))
    
    def save(self, filepath: str) -> None:
        data = {"name": self.name, "description": self.description, "phonemes": self.phonemes}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Voice '{self.name}' saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Voice':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        voice = cls(data['name'], data.get('description', ''))
        voice.phonemes = data['phonemes']
        return voice

class DefaultVoice(Voice):
    def __init__(self):
        super().__init__("Default", "Built-in robotic voice")
        self.phonemes = {
            'AH': {'f1': 700, 'f2': 1100, 'f3': 2400, 'f4': 115, 'length': 0.14, 'voiced': True},
            'AE': {'f1': 650, 'f2': 1250, 'f3': 2500, 'f4': 115, 'length': 0.14, 'voiced': True},
            'AA': {'f1': 620, 'f2': 1180, 'f3': 2550, 'f4': 115, 'length': 0.14, 'voiced': True},
            'AO': {'f1': 550, 'f2': 850, 'f3': 2400, 'f4': 115, 'length': 0.14, 'voiced': True},
            'EH': {'f1': 530, 'f2': 1700, 'f3': 2450, 'f4': 115, 'length': 0.14, 'voiced': True},
            'EY': {'f1': 400, 'f2': 2100, 'f3': 2800, 'f4': 115, 'length': 0.14, 'voiced': True},
            'IH': {'f1': 420, 'f2': 1950, 'f3': 2500, 'f4': 115, 'length': 0.14, 'voiced': True},
            'IY': {'f1': 300, 'f2': 2250, 'f3': 3000, 'f4': 115, 'length': 0.14, 'voiced': True},
            'OW': {'f1': 450, 'f2': 900, 'f3': 2350, 'f4': 115, 'length': 0.14, 'voiced': True},
            'UH': {'f1': 400, 'f2': 650, 'f3': 2400, 'f4': 115, 'length': 0.14, 'voiced': True},
            'UW': {'f1': 330, 'f2': 900, 'f3': 2200, 'f4': 115, 'length': 0.14, 'voiced': True},
            'ER': {'f1': 480, 'f2': 1180, 'f3': 1650, 'f4': 115, 'length': 0.14, 'voiced': True},
            'M':  {'f1': 350, 'f2': 1050, 'f3': 2250, 'f4': 115, 'length': 0.12, 'voiced': True},
            'N':  {'f1': 320, 'f2': 1150, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'NG': {'f1': 280, 'f2': 950, 'f3': 2350, 'f4': 115, 'length': 0.12, 'voiced': True},
            'L':  {'f1': 400, 'f2': 1150, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'R':  {'f1': 450, 'f2': 1250, 'f3': 1500, 'f4': 115, 'length': 0.12, 'voiced': True},
            'DH': {'f1': 380, 'f2': 1650, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'V':  {'f1': 380, 'f2': 1550, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'Z':  {'f1': 380, 'f2': 1750, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'ZH': {'f1': 380, 'f2': 1450, 'f3': 2250, 'f4': 115, 'length': 0.12, 'voiced': True},
            'W':  {'f1': 350, 'f2': 700, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'Y':  {'f1': 350, 'f2': 2050, 'f3': 2650, 'f4': 115, 'length': 0.12, 'voiced': True},
            'JH': {'f1': 400, 'f2': 1650, 'f3': 2450, 'f4': 115, 'length': 0.12, 'voiced': True},
            'B':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'D':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'G':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'P':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'T':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'K':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'F':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.125, 'voiced': False},
            'S':  {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.125, 'voiced': False},
            'SH': {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.125, 'voiced': False},
            'TH': {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.125, 'voiced': False},
            'HH': {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.125, 'voiced': False},
            'CH': {'f1': None, 'f2': None, 'f3': None, 'f4': 0, 'length': 0.068, 'voiced': False},
            'SIL': {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'length': 0.19, 'voiced': 'silence'},
        }

class VoiceRegistry:
    def __init__(self):
        self.voices: Dict[str, Voice] = {'Default': DefaultVoice()}
        self._load_custom_voices()
        self.current_voice: Voice = self.voices['Default']
    
    def _load_custom_voices(self):
        for filepath in VOICES_DIR.glob("*.json"):
            try:
                voice = Voice.load(filepath)
                self.voices[voice.name] = voice
            except Exception:
                pass
    
    def set_current_voice(self, name: str) -> bool:
        if name in self.voices:
            self.current_voice = self.voices[name]
            return True
        return False
    
    def list_voices(self) -> Dict[str, Voice]:
        return self.voices

VOICE_REGISTRY = VoiceRegistry()

def ipa_to_phoneme_sequence(ipa_text: str, for_spec: bool = False, voice: Voice = None) -> Union[List[str], List[Dict]]:
    """
    Convert IPA text to phoneme sequence with stress handling.
    for_spec=True: returns spec dicts with stress-modified duration/pitch
    for_spec=False: returns simple phoneme list (legacy format)
    American English IPA mapping with stress support (ˈ = primary, ˌ = secondary)
    """
    if voice is None:
        voice = VOICE_REGISTRY.current_voice
    
    # Normalize input: remove brackets/spaces, keep stress markers
    text = ipa_text.strip().replace('/', '').replace('[', '').replace(']', '')
    text = re.sub(r'\s+', '', text)  # Remove all spaces
    
    # IPA patterns ordered longest-first for proper tokenization
    patterns = [
        ('tʃ', 'CH'),
        ('dʒ', 'JH'),
        ('aɪ', 'EY'),   # /aɪ/ → EY approximation (best available)
        ('aʊ', 'OW'),   # /aʊ/ → OW approximation
        ('ɔɪ', 'OY_SEQ'), # Will expand to AO + IH sequence
        ('ʃ', 'SH'),
        ('ʒ', 'ZH'),
        ('θ', 'TH'),
        ('ð', 'DH'),
        ('ŋ', 'NG'),
        ('ˈ', 'PRIMARY_STRESS'),
        ('ˌ', 'SECONDARY_STRESS'),
        ('iː', 'IY'),
        ('uː', 'UW'),
        ('ɑː', 'AA'),
        ('ɔː', 'AO'),
        ('ɝ', 'ER'),
        ('ɚ', 'ER'),
        ('i', 'IY'),
        ('ɪ', 'IH'),
        ('eɪ', 'EY'),
        ('e', 'EY'),
        ('ɛ', 'EH'),
        ('æ', 'AE'),
        ('ɑ', 'AA'),
        ('ɔ', 'AO'),
        ('oʊ', 'OW'),
        ('o', 'OW'),
        ('ʊ', 'UH'),
        ('u', 'UW'),
        ('ʌ', 'AH'),
        ('ə', 'AH'),
        ('ɜ', 'ER'),
        ('p', 'P'),
        ('b', 'B'),
        ('t', 'T'),
        ('d', 'D'),
        ('k', 'K'),
        ('g', 'G'),
        ('m', 'M'),
        ('n', 'N'),
        ('l', 'L'),
        ('r', 'R'),
        ('f', 'F'),
        ('v', 'V'),
        ('s', 'S'),
        ('z', 'Z'),
        ('h', 'HH'),
        ('w', 'W'),
        ('j', 'Y'),
    ]
    
    i = 0
    phonemes: List[Tuple[str, int]] = []  # (phoneme, stress_level: 0=none,1=secondary,2=primary)
    pending_stress = 0
    
    while i < len(text):
        matched = False
        for pattern, mapping in patterns:
            if text.startswith(pattern, i):
                if mapping == 'PRIMARY_STRESS':
                    pending_stress = 2
                elif mapping == 'SECONDARY_STRESS':
                    pending_stress = 1
                elif mapping == 'OY_SEQ':
                    # Expand /ɔɪ/ to AO (stressed) + IH (unstressed)
                    phonemes.append(('AO', pending_stress if pending_stress else 0))
                    phonemes.append(('IH', 0))
                    pending_stress = 0
                elif mapping in VOWELS or mapping == 'ER':
                    phonemes.append((mapping, pending_stress))
                    pending_stress = 0
                else:
                    phonemes.append((mapping, 0))
                i += len(pattern)
                matched = True
                break
        if not matched:
            print(f"Warning: Unknown IPA character '{text[i]}' at position {i} - skipping")
            i += 1
    
    # Add SIL boundaries
    phonemes = [('SIL', 0)] + phonemes + [('SIL', 0)]
    
    if not for_spec:
        return [ph for ph, _ in phonemes]
    
    # Generate spec with stress-modified parameters
    specs = []
    for ph, stress in phonemes:
        base = voice.get_phoneme_data(ph)
        duration = base.get('length', 0.14)
        pitch_base = base.get('f4', 115.0) or 115.0
        
        # Apply stress modifiers (affects duration AND pitch contour)
        if stress == 2:  # Primary stress
            duration = min(duration * 1.55, 0.38)
            pitch_contour = [pitch_base * 1.42, pitch_base * 1.30]
        elif stress == 1:  # Secondary stress
            duration = min(duration * 1.30, 0.32)
            pitch_contour = [pitch_base * 1.18]
        else:  # Unstressed
            if ph == 'AH':  # Schwa reduction
                duration *= 0.82
            pitch_contour = [pitch_base * 0.88]
        
        f1 = base.get('f1', 0.0) or 0.0
        f2 = base.get('f2', 0.0) or 0.0
        f3 = base.get('f3', 0.0) or 0.0
        
        specs.append({
            'phoneme': ph,
            'duration': duration,
            'pitch_contour': pitch_contour,
            'num_pitch_points': len(pitch_contour),
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'voiced': ph not in {'SIL','B','D','G','P','T','K','F','S','SH','TH','HH','CH'}
        })
    
    return specs

def specs_to_readable(specs: List[Dict]) -> str:
    lines = ["# PHONEME DURATION P0 [P1 P2 ...]  (Stress affects duration/pitch)"]
    for spec in specs:
        pitches = ' '.join(f"{p:.1f}" for p in spec['pitch_contour'])
        lines.append(f"{spec['phoneme']:4s} {spec['duration']:6.3f} {pitches}")
    return '\n'.join(lines)

def menu_ipa_to_legacy_phoneme():
    print("\n(LEGACY) IPA to Phoneme")
    print("Convert American English IPA to simple phoneme list (for .phn format)")
    print("Stress markers (ˈˌ) are IGNORED in legacy output")
    print("Example IPA: hɛˈloʊ  →  HH EH L OW")
    
    choice = input("\nInput method: (1) file or (2) manual? [2]: ").strip() or '2'
    if choice == '1':
        filename = input("Enter IPA file (.txt): ").strip()
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return
        with open(filename, 'r', encoding='utf-8') as f:
            ipa_text = f.read().strip()
    else:
        print("\nEnter IPA text (e.g., hɛˈloʊ wɝld):")
        ipa_text = input("> ").strip()
        if not ipa_text:
            print("Error: Empty input!")
            return
    
    phonemes = ipa_to_phoneme_sequence(ipa_text, for_spec=False)
    
    if not phonemes or len(phonemes) <= 2:  # Only SIL boundaries
        print("Error: No valid phonemes parsed from IPA!")
        return
    
    print(f"\nConverted {len(phonemes)-2} phonemes (excluding SIL boundaries):")
    print("   " + " ".join(phonemes))
    
    save = input("\nSave to text file? (y/n) [n]: ").strip().lower()
    if save == 'y':
        default = "ipa_output.txt"
        fname = input(f"Filename [{default}]: ").strip() or default
        if not fname.endswith('.txt'):
            fname += '.txt'
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(" ".join(phonemes))
        print(f"Saved phoneme sequence to: {fname}")
        print("Use menu option 1 to convert this to .phn bytecode")

def menu_ipa_to_phoneme_spec():
    print("\nIPA to Phoneme Spec")
    print("Convert American English IPA to parameterized spec with STRESS handling")
    print("Primary stress (ˈ) → longer duration + rising pitch contour")
    print("Secondary stress (ˌ) → moderate duration + elevated pitch")
    print("Example IPA: ˈkʌmˌpjuːtɚ  →  stressed K AH M ... with pitch contours")
    
    choice = input("\nInput method: (1) file or (2) manual? [2]: ").strip() or '2'
    if choice == '1':
        filename = input("Enter IPA file (.txt): ").strip()
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return
        with open(filename, 'r', encoding='utf-8') as f:
            ipa_text = f.read().strip()
    else:
        print("\nEnter IPA text with stress markers (e.g., hɛˈloʊ ˈwɝld):")
        ipa_text = input("> ").strip()
        if not ipa_text:
            print("Error: Empty input!")
            return
    
    specs = ipa_to_phoneme_sequence(ipa_text, for_spec=True, voice=VOICE_REGISTRY.current_voice)
    
    if not specs or len(specs) <= 2:
        print("Error: No valid phonemes parsed from IPA!")
        return
    
    print(f"\nGenerated {len(specs)-2} parameterized phonemes (excluding SIL):")
    print(specs_to_readable(specs))
    
    save = input("\nSave spec to text file? (y/n) [y]: ").strip().lower() or 'y'
    if save == 'y':
        default = "ipa_spec.txt"
        fname = input(f"Filename [{default}]: ").strip() or default
        if not fname.endswith('.txt'):
            fname += '.txt'
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(specs_to_readable(specs))
        print(f"Saved spec to: {fname}")
        print("Use menu option 3 to convert this to .phx bytecode")

def text_to_phonemes(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    phoneme_sequence = ['SIL']
    for i, word in enumerate(words):
        phoneme_sequence.append('SIL')
    phoneme_sequence.append('SIL')
    return phoneme_sequence

def phonemes_to_spec(phonemes: List[str], voice: Voice, pitch_base: float = 115.0) -> List[Dict]:
    specs = []
    for i, ph in enumerate(phonemes):
        ph_data = voice.get_phoneme_data(ph)
        duration = ph_data.get('length', 0.14)
        if ph == 'SIL':
            pitch = [0.0]
        elif ph in VOWELS:
            pitch = [pitch_base]
        else:
            pitch = [pitch_base if ph_data.get('voiced', False) else 0.0]
        f1 = ph_data.get('f1', 0.0) or 0.0
        f2 = ph_data.get('f2', 0.0) or 0.0
        f3 = ph_data.get('f3', 0.0) or 0.0
        specs.append({
            'phoneme': ph,
            'duration': duration,
            'pitch_contour': pitch,
            'num_pitch_points': len(pitch),
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'voiced': ph not in {'SIL','B','D','G','P','T','K','F','S','SH','TH','HH','CH'}
        })
    return specs

def parse_phoneme_spec(text: str, voice: Voice) -> List[Dict]:
    specs = []
    for line_num, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        ph_name = parts[0].upper()
        if ph_name not in PHONEME_TO_BYTE:
            continue
        try:
            duration = max(0.01, min(2.0, float(parts[1])))
            pitch_points = [float(p) for p in parts[2:]]
            if len(pitch_points) > 8:
                pitch_points = pitch_points[:8]
            ph_data = voice.get_phoneme_data(ph_name)
            f1 = ph_data.get('f1', 0.0) or 0.0
            f2 = ph_data.get('f2', 0.0) or 0.0
            f3 = ph_data.get('f3', 0.0) or 0.0
            specs.append({
                'phoneme': ph_name,
                'duration': duration,
                'pitch_contour': pitch_points,
                'num_pitch_points': len(pitch_points),
                'f1': f1,
                'f2': f2,
                'f3': f3,
                'voiced': ph_name not in {'SIL','B','D','G','P','T','K','F','S','SH','TH','HH','CH'}
            })
        except ValueError:
            continue
    return specs

def save_parameterized_phonemes(filename: str, specs: List[Dict]):
    with open(filename, 'wb') as f:
        f.write(b'\xDE\xAD\xBE\xEF')
        for spec in specs:
            ph_id = PHONEME_TO_BYTE[spec['phoneme']]
            f.write(bytes([ph_id]))
            np.array([spec['duration']], dtype=np.float32).tofile(f)
            f.write(bytes([spec['num_pitch_points']]))
            pitches = spec['pitch_contour'] + [0.0] * (8 - spec['num_pitch_points'])
            np.array(pitches[:8], dtype=np.float32).tofile(f)
            np.array([spec['f1'], spec['f2'], spec['f3']], dtype=np.float32).tofile(f)
    print(f"Saved {len(specs)} parameterized phonemes to: {filename}")

def load_parameterized_phonemes(filename: str) -> List[Dict]:
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic != b'\xDE\xAD\xBE\xEF':
            raise ValueError("Not a valid PHX file")
        specs = []
        while True:
            ph_byte = f.read(1)
            if not ph_byte:
                break
            ph_id = ph_byte[0]
            if ph_id not in BYTE_TO_PHONEME:
                raise ValueError(f"Invalid phoneme ID: 0x{ph_id:02X}")
            dur_arr = np.fromfile(f, dtype=np.float32, count=1)
            if len(dur_arr) < 1:
                break
            duration = float(dur_arr[0])
            num_pts_byte = f.read(1)
            if not num_pts_byte:
                break
            num_pts = num_pts_byte[0]
            pitches_arr = np.fromfile(f, dtype=np.float32, count=8)
            if len(pitches_arr) < 8:
                break
            formants_arr = np.fromfile(f, dtype=np.float32, count=3)
            if len(formants_arr) < 3:
                break
            specs.append({
                'phoneme': BYTE_TO_PHONEME[ph_id],
                'duration': duration,
                'pitch_contour': [float(p) for p in pitches_arr[:num_pts]],
                'num_pitch_points': num_pts,
                'f1': float(formants_arr[0]),
                'f2': float(formants_arr[1]),
                'f3': float(formants_arr[2]),
                'voiced': BYTE_TO_PHONEME[ph_id] not in {'SIL','B','D','G','P','T','K','F','S','SH','TH','HH','CH'}
            })
    return specs

class FormantSynthesizer:
    def __init__(self, voice: Voice, sample_rate: int = smp):
        self.fs = sample_rate
        self.voice = voice
    
    def generate_glottal_pulse_train_contour(self, duration: float, pitch_contour: List[float]):
        n_samples = int(duration * self.fs)
        signal = np.zeros(n_samples)
        t = 0.0
        if not pitch_contour or all(p == 0 for p in pitch_contour):
            pitch_contour = [115.0]
        num_points = len(pitch_contour)
        while t < duration:
            t_norm = min(1.0, t / duration)
            if num_points == 1:
                f0 = pitch_contour[0]
            else:
                contour_pos = t_norm * (num_points - 1)
                idx_floor = int(contour_pos)
                frac = contour_pos - idx_floor
                if idx_floor >= num_points - 1:
                    f0 = pitch_contour[-1]
                else:
                    f0 = pitch_contour[idx_floor] * (1 - frac) + pitch_contour[idx_floor + 1] * frac
            f0 = max(50.0, min(400.0, f0))
            period_samples = self.fs / f0
            pulse_len = int(period_samples * 0.6)
            if pulse_len < 8:
                pulse_len = 8
            pulse = np.zeros(pulse_len)
            open_len = max(4, int(pulse_len * 0.4))
            pulse[:open_len] = -0.5 * (1 - np.cos(np.linspace(0, np.pi, open_len)))
            if pulse_len > open_len:
                close_len = pulse_len - open_len
                pulse[open_len:] = -0.1 * np.exp(-np.linspace(0, 5, close_len))
            start = int(t * self.fs)
            end = min(start + pulse_len, n_samples)
            if end > start:
                signal[start:end] += pulse[:end - start] * 0.6
            t += period_samples / self.fs
        peak = np.max(np.abs(signal))
        if peak > 0.1:
            signal = signal * (0.6 / peak)
        return signal
    
    def generate_shaped_noise(self, duration: float, phoneme: str, intensity: float = 0.25):
        n_samples = int(duration * self.fs)
        noise = np.random.randn(n_samples)
        if phoneme in {'S'}:
            b, a = sig.butter(6, [4000/(self.fs/2), 8500/(self.fs/2)], btype='band')
            noise = sig.filtfilt(b, a, noise)
            b2, a2 = sig.butter(4, 6500/(self.fs/2), btype='high')
            noise = sig.filtfilt(b2, a2, noise) * 1.3
        elif phoneme in {'SH', 'ZH'}:
            b, a = sig.butter(5, [2500/(self.fs/2), 6000/(self.fs/2)], btype='band')
            noise = sig.filtfilt(b, a, noise)
        elif phoneme in {'F', 'TH'}:
            b, a = sig.butter(4, 3500/(self.fs/2), btype='low')
            noise = sig.filtfilt(b, a, noise)
        elif phoneme == 'HH':
            b, a = sig.butter(3, 2800/(self.fs/2), btype='low')
            noise = sig.filtfilt(b, a, noise)
            noise += np.random.randn(n_samples) * 0.15
        elif phoneme in {'V', 'DH', 'Z'}:
            b, a = sig.butter(4, 4500/(self.fs/2), btype='low')
            noise = sig.filtfilt(b, a, noise)
            voicing = np.sin(2 * np.pi * 120 * np.arange(n_samples) / self.fs) * 0.15
            noise = noise * 0.85 + voicing * 0.15
        else:
            b, a = sig.butter(4, 7500/(self.fs/2), btype='low')
            noise = sig.filtfilt(b, a, noise)
        peak = np.max(np.abs(noise))
        if peak < 1e-6:
            noise = np.random.randn(n_samples) * intensity * 0.7
            peak = 1.0
        noise = noise * (intensity / peak)
        return noise[:n_samples]
    
    def stable_resonator(self, freq: float, bw: float):
        if freq <= 0:
            return np.array([1.0]), np.array([1.0])
        w0 = 2 * np.pi * freq / self.fs
        bw_rad = max(2 * np.pi * bw / self.fs, 2 * np.pi * 80 / self.fs)
        a1 = -2 * np.exp(-bw_rad/2) * np.cos(w0)
        a2 = np.exp(-bw_rad)
        b0 = np.sqrt(1 - a2)
        return np.array([b0]), np.array([1.0, a1, a2])
    
    def apply_formants_safe(self, signal: np.ndarray, f1: float, f2: float, f3: float) -> np.ndarray:
        b1, b2, b3 = 60, 90, 150
        for freq, bw in [(f1, b1), (f2, b2), (f3, b3)]:
            if freq and freq > 50:
                b, a = self.stable_resonator(freq, bw)
                signal = sig.lfilter(b, a, signal)
        peak = np.max(np.abs(signal))
        if peak > 4.0:
            signal = signal * (3.0 / peak)
        b, a = sig.butter(1, 900/(self.fs/2), btype='high')
        return sig.lfilter(b, a, signal)
    
    def synthesize_phoneme_direct(self, spec: Dict) -> np.ndarray:
        ph = spec['phoneme']
        dur = spec['duration']
        f1, f2, f3 = spec['f1'], spec['f2'], spec['f3']
        pitch_contour = spec['pitch_contour']
        voiced = spec['voiced']
        
        if ph == 'SIL':
            return np.zeros(int(dur * self.fs))
        
        if ph in STOPS:
            n_samples = int(dur * self.fs)
            out = np.zeros(n_samples)
            closure_end = int(n_samples * 0.82)
            burst_start = closure_end
            burst_len = min(200, n_samples - burst_start)
            if burst_len > 30:
                burst_noise = np.random.randn(burst_len)
                if f1 > 50:
                    b1, a1 = self.stable_resonator(f1, 150)
                    b2, a2 = self.stable_resonator(f2, 200)
                    burst_noise = sig.lfilter(b1, a1, burst_noise)
                    burst_noise = sig.lfilter(b2, a2, burst_noise)
                burst_env = np.hanning(burst_len) * 0.6
                out[burst_start:burst_start+burst_len] = burst_noise * burst_env
            if ph in {'P', 'T', 'K', 'CH'} and closure_end + burst_len < n_samples:
                aspir_start = burst_start + burst_len
                aspir_len = n_samples - aspir_start
                if aspir_len > 50:
                    aspiration = self.generate_shaped_noise(aspir_len/self.fs, 'HH', intensity=0.18)
                    b, a = sig.butter(2, 800/(self.fs/2), btype='high')
                    aspiration = sig.filtfilt(b, a, aspiration)
                    out[aspir_start:] = aspiration[:aspir_len] * 0.4
            elif ph in {'B', 'D', 'G'} and closure_end + burst_len < n_samples:
                voicing_start = burst_start + int(burst_len * 1.3)
                voicing_len = n_samples - voicing_start
                if voicing_len > 100:
                    voicing = self.generate_glottal_pulse_train_contour(voicing_len/self.fs, [115.0])
                    out[voicing_start:] = voicing[:voicing_len] * 0.35
            return out * 0.85
        
        if not voiced:
            if ph == 'S':
                intensity = 5.12
            elif ph == 'SH':
                intensity = 2.56
            elif ph in {'F', 'TH'}:
                intensity = 0.64
            else:
                intensity = 0.40
            source = self.generate_shaped_noise(dur, ph, intensity=intensity)
            if f1 > 50:
                if ph in {'S', 'SH'}:
                    source = self.apply_formants_safe(source, f1*0.7, f2*0.7, f3*0.7)
                else:
                    source = self.apply_formants_safe(source, f1, f2, f3)
            else:
                source = source * 0.45
            output = source
        else:
            source = self.generate_glottal_pulse_train_contour(dur, pitch_contour)
            if f1 > 50:
                output = self.apply_formants_safe(source, f1, f2, f3)
            else:
                output = source * 0.45
        
        n = len(output)
        env = np.ones(n)
        att = min(0.007, dur * 0.12)
        rel = min(0.018, dur * 0.28)
        att_s = int(att * self.fs)
        rel_s = int(rel * self.fs)
        if att_s > 0:
            env[:att_s] = np.linspace(0, 1, att_s)
        if rel_s > 0:
            env[-rel_s:] = np.linspace(1, 0.05, rel_s)
        output = output * env
        output = np.tanh(output * 1.15) * 0.93
        return output * 0.82
    
    def synthesize_from_specs(self, specs: List[Dict]) -> np.ndarray:
        segments = []
        total_duration = 0.0
        for i, spec in enumerate(specs):
            seg = self.synthesize_phoneme_direct(spec)
            segments.append(seg)
            total_duration += spec['duration']
            if spec['phoneme'] != 'SIL' and i < len(specs) - 1:
                next_ph = specs[i+1]['phoneme']
                if next_ph != 'SIL':
                    gap_dur = 0.003
                    segments.append(np.zeros(int(gap_dur * self.fs)))
                    total_duration += gap_dur
        audio = np.concatenate(segments)
        audio = np.tanh(audio * 1.25) * 0.94
        b, a = sig.butter(5, 5000/(self.fs/2), btype='low')
        audio = sig.filtfilt(b, a, audio)
        return audio

def save_wav(filename: str, audio: np.ndarray, sr: int = smp):
    audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

def menu_legacy_phoneme_to_bytecode():
    print("\n(LEGACY) PHONEME -> BYTECODE (.phn)")
    print("Convert readable phonemes (SIL HH EH L OW) to simple bytecode (.phn)")
    print("Format: One byte per phoneme (0x00-0xFF) - NO pitch contours")
    
    choice = input("\nInput method: (1) file or (2) manual? [2]: ").strip() or '2'
    if choice == '1':
        filename = input("Enter phoneme file (.txt): ").strip()
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        phonemes = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            for token in parts:
                token = token.strip().upper()
                if token in PHONEME_TO_BYTE:
                    phonemes.append(token)
                elif re.match(r'^-?\d+(\.\d+)?$', token):
                    continue
    else:
        print("\nEnter phonemes (space-separated, e.g., SIL HH EH L OW):")
        text = input("> ").strip()
        if not text:
            print("Error: Empty input!")
            return
        phonemes = []
        for token in text.upper().split():
            if token in PHONEME_TO_BYTE:
                phonemes.append(token)
            elif re.match(r'^-?\d+(\.\d+)?$', token):
                continue
    
    if not phonemes:
        print("Error: No valid phonemes!")
        return
    
    print(f"\nValidated {len(phonemes)} phonemes:")
    print("   " + " ".join(phonemes))
    
    default_name = "output.phn"
    filename = input(f"\nSave as [{default_name}]: ").strip() or default_name
    if not filename.endswith('.phn'):
        filename += '.phn'
    
    voice_name = VOICE_REGISTRY.current_voice.name
    voice_bytes = voice_name.encode('utf-8')
    with open(filename, 'wb') as f:
        f.write(b'\xFE\xEB\xDA\xED')
        f.write(bytes([len(voice_bytes)]))
        f.write(voice_bytes)
        for ph in phonemes:
            f.write(bytes([PHONEME_TO_BYTE[ph]]))
    
    print(f"Saved {len(phonemes)} phonemes to: {filename}")

def menu_legacy_bytecode_to_audio():
    print("\nLEGACY BYTECODE -> AUDIO (.phn)")
    print("Format: FE EB DA ED header + voice name + 1 byte per phoneme")
    
    filename = input("\nEnter legacy bytecode file (.phn): ").strip()
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return
    
    with open(filename, 'rb') as f:
        content = f.read()
    
    if content.startswith(b'\xFE\xEB\xDA\xED') and len(content) > 5:
        name_len = content[4]
        try:
            voice_name = content[5:5+name_len].decode('utf-8')
            byte_data = content[5+name_len:]
            if VOICE_REGISTRY.set_current_voice(voice_name):
                print(f"Using voice from file: {voice_name}")
        except:
            byte_data = content[4+1:]
    else:
        byte_data = content
    
    print(f"Loaded {len(byte_data)} phonemes from {filename}")
    phonemes = []
    for byte_val in byte_:
        if byte_val in BYTE_TO_PHONEME:
            phonemes.append(BYTE_TO_PHONEME[byte_val])
    
    if not phonemes:
        print("Error: No valid phonemes!")
        return
    
    specs = []
    for ph in phonemes:
        ph_data = VOICE_REGISTRY.current_voice.get_phoneme_data(ph)
        duration = ph_data.get('length', 0.14)
        pitch = 115.0 if ph_data.get('voiced', False) and ph != 'SIL' else 0.0
        f1 = ph_data.get('f1', 0.0) or 0.0
        f2 = ph_data.get('f2', 0.0) or 0.0
        f3 = ph_data.get('f3', 0.0) or 0.0
        specs.append({
            'phoneme': ph,
            'duration': duration,
            'pitch_contour': [pitch],
            'num_pitch_points': 1,
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'voiced': ph not in {'SIL','B','D','G','P','T','K','F','S','SH','TH','HH','CH'}
        })
    
    synth = FormantSynthesizer(VOICE_REGISTRY.current_voice, sample_rate=smp)
    audio = synth.synthesize_from_specs(specs)
    
    default_wav = Path(filename).stem + ".wav"
    wav_name = input(f"\nSave WAV as [{default_wav}]: ").strip() or default_wav
    if not wav_name.endswith('.wav'):
        wav_name += '.wav'
    
    save_wav(wav_name, audio, smp)

def menu_spec_to_bytecode():
    print("\nPHONEME SPEC -> BYTECODE (.phx)")
    print("Format: PHONEME DURATION P0 [P1 P2 ... Pn]  (pitch in Hz)")
    
    choice = input("\nRead from (1) file or (2) manual input? [1]: ").strip() or '1'
    if choice == '1':
        filename = input("Enter phoneme spec filename (.txt): ").strip()
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("\nEnter phoneme specs (empty line to finish):")
        lines = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            lines.append(line)
        text = '\n'.join(lines)
    
    specs = parse_phoneme_spec(text, VOICE_REGISTRY.current_voice)
    if not specs:
        print("Error: No valid phonemes parsed!")
        return
    
    print(f"\nParsed {len(specs)} phonemes:")
    print(specs_to_readable(specs))
    
    default_name = "output.phx"
    filename = input(f"\nSave as [{default_name}]: ").strip() or default_name
    if not filename.endswith('.phx'):
        filename += '.phx'
    
    save_parameterized_phonemes(filename, specs)

def menu_new_bytecode_to_audio():
    print("\nBYTECODE -> AUDIO (.phx)")
    print("Format: PHX header (DE AD BE EF) + parameterized phoneme data")
    
    filename = input("\nEnter parameterized bytecode file (.phx): ").strip()
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return
    
    try:
        specs = load_parameterized_phonemes(filename)
    except Exception as e:
        print(f"Error loading .phx file: {e}")
        return
    
    if not specs:
        print("Error: No valid phonemes loaded!")
        return
    
    print(f"\nLoaded {len(specs)} parameterized phonemes:")
    print(specs_to_readable(specs))
    
    choice = input("\nSynthesize to WAV now? (y/n) [y]: ").strip().lower()
    if choice in ('', 'y'):
        synth = FormantSynthesizer(VOICE_REGISTRY.current_voice, sample_rate=smp)
        audio = synth.synthesize_from_specs(specs)
        default_wav = Path(filename).stem + ".wav"
        wav_name = input(f"Save WAV as [{default_wav}]: ").strip() or default_wav
        if not wav_name.endswith('.wav'):
            wav_name += '.wav'
        save_wav(wav_name, audio, smp)
        
        choice = input(f"Play audio now? (y/n) [n]: ").strip().lower()
        if choice == 'y':
            try:
                if os.name == 'nt':
                    os.startfile(wav_name)
                elif sys.platform == 'darwin':
                    subprocess.run(['afplay', wav_name])
                else:
                    subprocess.run(['aplay', wav_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("Playing audio...")
            except Exception as e:
                print(f"Could not play audio automatically: {e}")

def menu_voice_management():
    while True:
        print("\nVOICE MANAGEMENT MENU")
        print("1. List available voices")
        print("2. Choose voice")
        print("3. Back to main menu")
        
        choice = input("\nSelect option (1-3): ").strip()
        if choice == '1':
            print("\nAvailable voices:")
            for name, voice in VOICE_REGISTRY.list_voices().items():
                marker = " <- CURRENT" if name == VOICE_REGISTRY.current_voice.name else ""
                print(f"  * {name:20s} : {voice.description}{marker}")
        elif choice == '2':
            name = input("Enter voice name: ").strip()
            if VOICE_REGISTRY.set_current_voice(name):
                print(f"Voice changed to: {name}")
            else:
                print(f"Voice '{name}' not found")
        elif choice == '3':
            break
        else:
            print("Invalid option. Please enter 1-3.")
        
        input("\nPress Enter to continue...")

def main_menu():
    while True:
        print("\nFORMANT SYNTHESIS TTS - CLI MODE")
        print("1. (LEGACY) Phoneme -> Bytecode (.phn)")
        print("2. (LEGACY) Bytecode -> Audio (.phn)")
        print("3. (LEGACY) IPA to Phoneme")
        print("4. IPA to Phoneme Spec  (with stress handling)")
        print("5. Phoneme Spec -> Bytecode (.phx)")
        print("6. Bytecode -> Audio (.phx)")
        print("7. Voice Management")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        if choice == '1':
            menu_legacy_phoneme_to_bytecode()
        elif choice == '2':
            menu_legacy_bytecode_to_audio()
        elif choice == '3':
            menu_ipa_to_legacy_phoneme()
        elif choice == '4':
            menu_ipa_to_phoneme_spec()
        elif choice == '5':
            menu_spec_to_bytecode()
        elif choice == '6':
            menu_new_bytecode_to_audio()
        elif choice == '7':
            menu_voice_management()
        elif choice == '8':
            print("\nGoodbye! Happy synthesizing!")
            break
        else:
            print("Invalid option. Please enter 1-8.")
        
        input("\nPress Enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')

def cli_mode():
    parser = argparse.ArgumentParser(
        description='Formant Synthesis TTS - Minimal CLI',
        epilog='Examples:\n  %(prog)s --ipa "hɛˈloʊ" -o hello.wav\n  %(prog)s --text "hello world" --voice Default',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--text', metavar='TEXT', help='Synthesize English text (basic mapping)')
    group.add_argument('--ipa', metavar='IPA', help='Synthesize from IPA string with stress markers (e.g., hɛˈloʊ)')
    parser.add_argument('-o', '--output', metavar='FILE', help='Output WAV filename (default: auto-generated)')
    parser.add_argument('--voice', metavar='NAME', default='Default', help='Voice name (default: Default)')
    parser.add_argument('--list-voices', action='store_true', help='List available voices and exit')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed processing steps')
    args = parser.parse_args()

    # Voice listing mode
    if args.list_vvoices:
        print("Available voices:")
        for name, voice in VOICE_REGISTRY.list_voices().items():
            marker = " (current)" if name == VOICE_REGISTRY.current_voice.name else ""
            print(f"  {name}: {voice.description}{marker}")
        return

    # Set voice with validation
    if args.verbose:
        print(f"Requested voice: {args.voice}")
    if not VOICE_REGISTRY.set_current_voice(args.voice):
        print(f"Warning: Voice '{args.voice}' not found. Available voices:")
        for name in VOICE_REGISTRY.list_voices().keys():
            print(f"  - {name}")
        print("Falling back to 'Default' voice.")
        VOICE_REGISTRY.set_current_voice('Default')
    voice = VOICE_REGISTRY.current_voice
    if args.verbose:
        print(f"Using voice: {voice.name}")

    # No arguments = interactive mode
    if not any([args.text, args.ipa, args.list_voices]):
        if args.verbose:
            print("No synthesis arguments provided. Entering interactive menu mode.")
        main_menu()
        return

    synth = FormantSynthesizer(voice, sample_rate=smp)
    specs = None
    source_desc = ""

    # Parse input
    try:
        if args.ipa:
            if args.verbose:
                print(f"Parsing IPA: {args.ipa}")
            specs = ipa_to_phoneme_sequence(args.ipa, for_spec=True, voice=voice)
            source_desc = f"IPA '{args.ipa}'"
        elif args.text:
            if args.verbose:
                print(f"Converting text: {args.text}")
            phonemes = text_to_phonemes(args.text)
            specs = phonemes_to_spec(phonemes, voice)
            source_desc = f"text '{args.text}'"
    except Exception as e:
        print(f"ERROR parsing input: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Validate output
    if not specs or len(specs) <= 2:  # Only SIL boundaries
        print(f"ERROR: No valid phonemes generated from {source_desc}")
        sys.exit(1)

    # Show stats
    total_duration = sum(spec['duration'] for spec in specs)
    phoneme_count = len(specs) - 2  # exclude SIL boundaries
    if args.verbose:
        print(f"Parsed {phoneme_count} phonemes")
        print(f"Estimated duration: {total_duration:.2f}s")
        if args.ipa:
            print("Phoneme sequence:")
            for spec in specs:
                if spec['phoneme'] != 'SIL':
                    stress_note = f" (stress: {len(spec['pitch_contour'])} pts)" if len(spec['pitch_contour']) > 1 else ""
                    print(f"  {spec['phoneme']:4s} {spec['duration']:5.3f}s{stress_note}")
    else:
        print(f"Synthesizing {phoneme_count} phonemes ({total_duration:.2f}s) using '{voice.name}' voice...")

    # Synthesize
    try:
        import time
        start = time.time()
        audio = synth.synthesize_from_specs(specs)
        elapsed = time.time() - start
        if args.verbose:
            print(f"Synthesis completed in {elapsed:.3f}s ({total_duration/elapsed:.1f}x real-time)")
    except Exception as e:
        print(f"ERROR during synthesis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Determine output filename
    if not args.output:
        if args.ipa:
            base = re.sub(r'[^a-z0-9]', '_', args.ipa.lower().replace('ˈ', '').replace('ˌ', ''))[:24]
        else:
            base = re.sub(r'[^a-z0-9]', '_', args.text.lower())[:24]
        args.output = base + ".wav"
    if not args.output.endswith('.wav'):
        args.output += '.wav'

    # Save
    try:
        save_wav(args.output, audio, synth.fs)
        size_kb = os.path.getsize(args.output) / 1024
        print(f"Saved to: {args.output} ({size_kb:.1f} KB, {len(audio)/smp:.2f}s)")
    except Exception as e:
        print(f"ERROR saving WAV: {e}")
        sys.exit(1)

    # Playback hint
    if not args.verbose:
        print("Done.")
    else:
        print("\nPlayback hints:")
        if os.name == 'nt':
            print("  Windows: start", args.output)
        elif sys.platform == 'darwin':
            print("  macOS:   afplay", args.output)
        else:
            print("  Linux:   aplay", args.output)

if __name__ == "__main__":
    try:
        cli_mode()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
