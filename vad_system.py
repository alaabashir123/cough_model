import os
import numpy as np
from pydub import AudioSegment, silence
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa                 
import librosa.display         

class SystematicVAD:
    def __init__(self, target_folder):
        self.target_folder = target_folder
        # HUMAN-CENTRIC CONSTANTS
        self.MIN_SOUND_LEN = 100       # ms: Minimum duration of a sound peak
        self.BREATH_BRIDGE = 600       # ms: Bridges gaps < 600ms (intra-cough breath)
        self.PRE_PADDING = 200  # 0.2s before the sound
        self.POST_PADDING = 300 # 0.3s after the sound

        # --- BIO-FILTER LIMITS ---
        self.MIN_BIO_DUR = 0.2      # seconds
        self.MAX_BIO_DUR = 3.5      # seconds

        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
    
    def process_file(self, path):
        audio_filename = os.path.splitext(os.path.basename(path))[0]
        signal_pydub = AudioSegment.from_wav(path)
        sr, signal_array = wavfile.read(path)

        peak = signal_pydub.max_dBFS
        if peak == float('-inf'): return 0, []

        # 1. MEASURE THE HISS (Noise Floor)
        samples = np.array(signal_pydub.get_array_of_samples())
        p30 = np.percentile(np.abs(samples), 30)
        if p30 > 0:
            floor_db = 20 * np.log10(p30 / 32768)
        else:
            floor_db = -60
        # 2. CALCULATE THE WINDOW (The space between noise and cough)
        gap = peak - floor_db
        
        # 3. THE SNR-ADAPTIVE THRESHOLD
        if gap > 12:
            # High quality file: sit 6dB above noise
            thresh = floor_db + 6.0
        else:
            # Low quality/Quiet file: sit exactly in the middle of the gap
            thresh = floor_db + (gap / 2)

        # 4. SAFETY CLAMPS (Generalization)
        # Never go quieter than -55dB (for extremely quiet files like 014)
        thresh = max(thresh, -55)
        # Never let thresh get closer than 3dB to the peak
        thresh = min(thresh, peak - 3.0)

        # print(f"--- VAD: {audio_filename} | Peak: {round(peak, 2)} | Thresh: {round(thresh, 2)} ---")

        # 2. Find Sound Islands
        islands = silence.detect_nonsilent(signal_pydub, 
                                           min_silence_len=self.MIN_SOUND_LEN, 
                                           silence_thresh=thresh)

        # 3. Bridge the Islands (The "Breath Bridge")
        merged = []
        if islands:
            curr_start, curr_end = islands[0]
            for next_start, next_end in islands[1:]:
                if next_start - curr_end <= self.BREATH_BRIDGE:
                    curr_end = next_end
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        # 4. Apply Padding and Extract
        final_segments = []
        glance_metadata = []
        count = 0
        for start_ms, end_ms in merged:
            pad_start_ms = max(0, start_ms - self.PRE_PADDING)
            pad_end_ms = min(len(signal_pydub), end_ms + self.POST_PADDING)            


            # 2. THE SLIDING WINDOW FIX:
            # If this sound island is longer than 5 seconds (5000ms), 
            # we loop through it and cut it into 5-second pieces.
            current_pointer = pad_start_ms

            while current_pointer < pad_end_ms:
                # Calculate the end of this specific 5s chunk
                chunk_end_ms = min(current_pointer + 5000, pad_end_ms)
                
                # Calculate duration of this piece
                duration_sec = (chunk_end_ms - current_pointer) / 1000
                
                # Bio-Filter: Don't save tiny leftover scraps (e.g. 0.05s)
                if duration_sec >= self.MIN_BIO_DUR:
                    start_sample = int((current_pointer / 1000) * sr)
                    end_sample = int((chunk_end_ms / 1000) * sr)
                    chunk = signal_array[start_sample:end_sample]

                    if len(chunk) > 0:
                        timestamp = current_pointer / 1000
                        out_name = f"{audio_filename}_seg{count}_{format(timestamp, '.3f')}.wav"
                        wavfile.write(os.path.join(self.target_folder, out_name), sr, chunk)
                        
                        # Store metadata for the final audit plot
                        glance_metadata.append((timestamp, chunk_end_ms / 1000))
                        count += 1
                
                # Move the pointer forward 5 seconds
                current_pointer += 5000

        return count, glance_metadata