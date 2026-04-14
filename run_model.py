import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
import shutil
import numpy as np
from pydub import AudioSegment

# Custom Clinical Modules
from vad_system import SystematicVAD
from YAMNet_Interface import YAMNetClassifier
from refiner import AcousticRefiner

def clear_workspace():
    """Ensures a clean environment for the clinical audit."""
    folders = ['vad_audio', 'cough_monitor_results']
    for folder in folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except PermissionError:
                print(f"!!! Warning: {folder} is in use. Clearing files individually...")
                for f in os.listdir(folder):
                    try: os.remove(os.path.join(folder, f))
                    except: pass
        os.makedirs(folder, exist_ok=True)
    
    # Remove old tracking files
    for f in ['./coughsegs.csv', './detections.csv']:
        if os.path.exists(f): 
            try: os.remove(f)
            except: pass

def sanitize_and_rename(target_path):
    """Clean the dataset: Convert FLAC to WAV, delete silent files, standardize names."""
    print(f"\n--- Stage 0: Sanitizing Dataset in {target_path} ---")
    
    # 1. Look for .wav and .flac files
    files = [f for f in os.listdir(target_path) if f.lower().endswith(('.wav', '.flac'))]
    
    if not files:
        print("!!! No audio files found in directory.")
        return

    counts = {}
    for filename in tqdm(files, desc="Sanitizing"):
        path = os.path.join(target_path, filename)
        try:
            # 2. Load the audio (pydub handles FLAC automatically IF FFmpeg is installed)
            audio = AudioSegment.from_file(path)
            
            # 3. Standardization: Peak Volume Check
            if audio.max_dBFS < -55:
                print(f"Deleting {filename}: File is too quiet.")
                os.remove(path)
                continue

            # 4. Standardize naming (cough_heavy_001.wav, etc.)
            intent = "misc"
            name_lower = filename.lower()
            for tag in ['cough', 'breathing', 'vowel', 'counting']:
                if tag in name_lower:
                    intent = tag
                    if 'heavy' in name_lower: intent += "_heavy"
                    if 'shallow' in name_lower: intent += "_shallow"
                    break
            
            counts[intent] = counts.get(intent, 0) + 1
            new_name = f"{intent}_{str(counts[intent]).zfill(3)}.wav"
            new_path = os.path.join(target_path, new_name)

            # 5. Convert to WAV and export
            audio.export(new_path, format="wav")
            
            # 6. Cleanup: If the original was a .flac or had a different name, delete it
            if os.path.abspath(path) != os.path.abspath(new_path):
                os.remove(path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def run_clinical_pipeline(target_path):
    # 0. Setup
    sanitize_and_rename(target_path)
    audio_files = sorted([f for f in os.listdir(target_path) if f.lower().endswith('.wav')])
    clear_workspace()
    # target_path = os.path.abspath(target_path)
    
    # 1. Initialize Engines
    print("--- Initializing Clinical Pipeline ---")
    vad = SystematicVAD(target_folder='vad_audio')
    classifier = YAMNetClassifier(model_path='audio_to_cough.tflite')
    auditor = AcousticRefiner()
    
    # 2. VAD Stage (Segmentation)
    audio_files = sorted([f for f in os.listdir(target_path) if f.lower().endswith('.wav')])    
    if not audio_files:
        print("No audio files found!")
        return
    print(f"\nStage 1: Segmenting {len(audio_files)} files...")
    for f in tqdm(audio_files, desc="Temporal Slicing"):
        vad.process_file(os.path.join(target_path, f))

    # 3. Stage 2 (CR-Fusion) & Stage 3 (Physics Audit)
    vad_segments = sorted([f for f in os.listdir('vad_audio') if f.endswith(('.wav'))])
    segment_data = []

    if not vad_segments:
        print("!!! WARNING: No audio segments detected by VAD. Audio may be too quiet or thresholds too strict.")
        # Create an empty DataFrame with the expected columns to avoid KeyError
        results_df = pd.DataFrame(columns=["OriginalFile", "SegmentPath", "Fused_Prob", "TopClass", "IsValid"])
    else:
        print(f"\nStage 2 & 3: AI Inference + Physics Audit on {len(vad_segments)} segments...")
        for seg_name in tqdm(vad_segments, desc="Processing"):
            sr, data = wavfile.read(os.path.join('vad_audio', seg_name))
            
            # Ensure data is 1D (Mono)
            if len(data.shape) > 1: data = np.mean(data, axis=1)

            # --- STAGE 2: CR-FUSION SCREENING ---
            # Returns Texture Prob, Physics Prob, and Verdict Level
            res = classifier.predict_segment(data, sr, seg_name)
            
            # --- STAGE 3: MICRO-PHYSICS AUDIT ---
            # Checks ECR/ZCR of the peak if Stage 1 flagged it as a candidate
            # Only run the auditor if the AI flagged it as 1 (Accepted or Suspect)
            if res["preds"] == 1:
                valid_peaks = auditor.count_and_audit_peaks(data, res["verdict_level"])
            else:
                valid_peaks = [] # Ensure it's empty if Stage 1 said No
            peak_count = len(valid_peaks)
            # Final Clinical Validation Logic
            is_validated_cough = 1 if (res["preds"] == 1 and peak_count) else 0
            
            segment_data.append({
                "OriginalFile": seg_name.split('_seg')[0] + ".wav",
                "SegmentPath": seg_name,
                "Fused_Prob": res["probs"],
                "TopClass": res["top_class"],
                "Audit": "Passed" if peak_count else "Failed",
                "IsValid": is_validated_cough
            })

    # 4. FINAL REPORT GENERATION
    results_df = pd.DataFrame(segment_data)
    # results_df.to_csv('cough_monitor_results/segment_details.csv', index=False)
    
    report_path = 'cough_monitor_results/clinical_audit_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== CLINICAL COUGH QUALITY AUDIT (CR-FUSION) ===\n")
        f.write("Criteria: 3 Validated Segments = GOOD AUDIO\n")
        f.write("-" * 50 + "\n\n")

        unique_files = results_df['OriginalFile'].unique()
        for filename in unique_files:
            file_group = results_df[results_df['OriginalFile'] == filename]
            valid_count = file_group['IsValid'].sum()
            
            # THE CLINICAL VERDICT
            quality = "GOOD AUDIO" if valid_count == 3 else "BAD AUDIO"
            
            f.write(f"FILE: {filename}\n")
            f.write(f"  >> Quality Verdict: {quality}\n")
            f.write(f"  >> Validated Cough Segments: {valid_count}\n")
            
            if valid_count > 0:
                # Find segments that passed
                hits = file_group[file_group['IsValid'] == 1]
                timestamps = [p.replace('.wav', '').rsplit('_', 1)[1] for p in hits['SegmentPath']]
                # f.write(f"  >> Cough Events detected at: {', '.join(timestamps)}s\n")
            f.write("-" * 50 + "\n")

    return report_path

if __name__ == "__main__":
    # Ensure script runs from its own directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help="Folder containing raw WAV files")
    args = parser.parse_args()
    
    if os.path.exists(args.folder):
        final_path = run_clinical_pipeline(args.folder)
        # print(f"\nAUDIT COMPLETE.")
        # print(f"Clinical Report: {final_path}")
        # print(f"Detailed CSV: cough_monitor_results/segment_details.csv")
    else:
        print(f"Error: Folder {args.folder} not found.")
    
    if os.path.exists('vad_audio'):
        shutil.rmtree('vad_audio')
        print("--- Workspace Cleaned: temporary VAD files deleted ---")