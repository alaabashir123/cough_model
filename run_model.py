import os
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
import shutil

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

def run_clinical_pipeline(target_path):
    # 0. Setup
    clear_workspace()
    target_path = os.path.abspath(target_path)
    
    # 1. Initialize Engines
    print("--- Initializing Clinical Pipeline ---")
    vad = SystematicVAD(target_folder='vad_audio')
    classifier = YAMNetClassifier(model_path='cough_cr_fusion_expert.tflite')
    auditor = AcousticRefiner()
    
    # 2. VAD Stage (Segmentation)
    audio_files = sorted([f for f in os.listdir(target_path) if f.endswith('.wav')])
    print(f"\nStage 1: Segmenting {len(audio_files)} files...")
    for f in tqdm(audio_files, desc="Temporal Slicing"):
        vad.process_file(os.path.join(target_path, f))

    # 3. Stage 2 (CR-Fusion) & Stage 3 (Physics Audit)
    vad_segments = sorted([f for f in os.listdir('vad_audio') if f.endswith('.wav')])
    segment_data = []

    # print(f"\nStage 2 & 3: CR-Fusion Inference + Physics Audit...")
    for seg_name in tqdm(vad_segments, desc="Processing"):
        sr, data = wavfile.read(os.path.join('vad_audio', seg_name))
        
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
    results_df.to_csv('cough_monitor_results/segment_details.csv', index=False)
    
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
        print(f"Clinical Report: {final_path}")
        print(f"Detailed CSV: cough_monitor_results/segment_details.csv")
    else:
        print(f"Error: Folder {args.folder} not found.")