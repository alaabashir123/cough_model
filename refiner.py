import numpy as np
import scipy.signal as sci

class AcousticRefiner:
    """
    Mobile-Ready Stage 2: Pure Physics Auditor.
    """
    def __init__(self):
        # --- CLINICAL GOLDILOCKS ZONES ---
        self.MIN_CREST = 2.8   # Sharpness floor
        self.MAX_CREST = 15.0  # Mechanical noise ceiling
        self.MIN_ECR = 0.35    # Energy concentration (Tightness)
        self.MIN_ZCR = 0.05    # Air turbulence (The Speech Killer)
        self.SR = 16000
    
    def count_and_audit_peaks(self, wav_data, verdict_level):
        # 1. STANDARDIZE & FLATTEN
        wav_data = np.array(wav_data).flatten().astype(np.float32)
        
        # ---SCALE CHECK ---
        # If data is in 16-bit integers (max > 1.0), scale to float
        if np.max(np.abs(wav_data)) > 1.0:
            wav_data = wav_data / 32768.0

        envelope = np.abs(wav_data)
        data_len = len(envelope)
        
        # --- BULLETPROOF SAVGOL LOGIC ---
        win_len = 1001 
        # Requirement 1: Window must be smaller than data
        if win_len >= data_len:
            win_len = data_len - 1 if data_len % 2 == 0 else data_len
        # Requirement 2: Window must be odd
        if win_len % 2 == 0:
            win_len -= 1
        # Requirement 3: Window must be greater than polynomial order (2)
        if win_len < 3:
            # Chunk is too small to smooth, use raw envelope
            envelope_smooth = envelope
        else:
            try:
                envelope_smooth = sci.savgol_filter(envelope, win_len, 2)
            except Exception:
                # Last resort fallback if scipy still complains
                envelope_smooth = envelope
        # ---------------------------------

        # 2. FIND CANDIDATE PEAKS
        # distance=3200 ensures 200ms separation at 16kHz
        mean_vol = np.mean(envelope_smooth)
        # --- SILENCE GUARD ---
        # If the whole segment is basically silent, return empty list
        if mean_vol < 1e-5: 
            return []
        # min_h: Based on mean volume
        min_h = mean_vol * (1.5 if int(verdict_level) == 1 else 2.5)
        peaks, _ = sci.find_peaks(envelope_smooth, height=min_h, distance=3200)
        
        valid_timestamps = []
        for p in peaks:
            # --- PATH A: FAST-TRACK (Level 1) ---
            if int(verdict_level) == 1:
                valid_timestamps.append(round(p / 16000, 3))
                continue

            # --- PATH B: BIOLOGICAL AUDIT (Level 2) ---
            start = max(0, p - 1600)
            end = min(data_len, p + 1600)
            window = wav_data[start:end]

            rms_val = np.sqrt(np.mean(window**2)) + 1e-6
            peak_val = np.max(np.abs(window))

            # If the window is near-silent, it's not a cough
            if rms_val < 1e-4:
                continue

            # Use a safe denominator for division
            safe_rms = rms_val + 1e-9
            crest = peak_val / safe_rms
            mid = len(window) // 2
            core = window[max(0, mid-480) : min(len(window), mid+480)]
            ecr = (np.sqrt(np.mean(core**2)) + 1e-6) / safe_rms
            zcr = np.mean(np.abs(np.diff(np.sign(window))) > 0)

            # 2-out-of-3 Point System
            score = 0
            if self.MIN_CREST < crest < self.MAX_CREST: score += 1
            if ecr > self.MIN_ECR: score += 1
            if zcr > self.MIN_ZCR: score += 1

            # Clinical Rescue
            if (score >= 2) or (crest > 7 and ecr > 1.0):
                valid_timestamps.append(round(p / 16000, 3))
                # print(f"Audit Accepted: Crest:{round(crest,1)} ECR:{round(ecr,2)} ZCR:{round(zcr,2)}")
            # else:
                # print(f"Audit Rejected: Crest:{round(crest,1)} ECR:{round(ecr,2)} ZCR:{round(zcr,2)}")
        return valid_timestamps