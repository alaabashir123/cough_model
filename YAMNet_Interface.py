import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.signal as signal
import csv
import os

class YAMNetClassifier:
    """
    Stage 1 Clinical Specialist:
    Combines Deep Learning (YAMNet) with Bio-Acoustic Physics.
    Input: Raw audio data.
    Output: 1035-feature classification (Embeddings + Physics + Veto Classes).
    """
    def __init__(self, model_path='audio_to_cough.tflite'):
        print("--- Initializing Clinical 1035 YAMNet Specialist ---")
        
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        # 1. Load YAMNet (The Feature Extractor)
        self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=None)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]['index']
        self.output_details = self.interpreter.get_output_details()
        
        # 2. Clinical Veto Indices (Indices for AudioSet classes)
        # 0:Speech, 16:Laughter, 57:HandClap, 69:Cough, 70:ThroatClearing, 
        # 71:Sneeze, 72:Sniff, 303:DoorSlam
        self.veto_indices = [0, 16, 57, 69, 70, 71, 72, 303]
        self._setup_class_names()

    def _setup_class_names(self):
        """Loads AudioSet display names for debugging/reporting."""
        class_map_path = self.yamnet.class_map_path().numpy()
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            self.class_names = [row['display_name'] for row in reader]

    def predict_segment(self, wav_data, sr, segment_tag):
        """
        Builds the 1035-feature vector and returns a clinical verdict.
        """
        # --- 1. SIGNAL STANDARDIZATION ---
        wav_data = np.array(wav_data).astype(np.float32)
        # Peak Normalization to range [-1, 1]
        max_abs = np.max(np.abs(wav_data))
        if max_abs > 1.0:
            wav_data = wav_data / (max_abs + 1e-8)
        
        # Mix to Mono and Flatten
        if len(wav_data.shape) > 1: 
            wav_data = np.mean(wav_data, axis=1)
        wav_data = wav_data.flatten()
        
        # Force 16kHz (YAMNet requirement)
        if sr != 16000:
            num_samples = int(len(wav_data) * 16000 / sr)
            wav_data = signal.resample(wav_data, num_samples)
            sr = 16000

        # --- 2. VOLUME GATE (THE SILENCE FILTER) ---
        rms = np.sqrt(np.mean(wav_data**2))
        if rms < 0.001 or max_abs < 0.01:
            return {
                "preds": 0, "probs": 0.0, "verdict_level": 0, 
                "top_class": "Silence/LowVol", "status": "Filtered (Silence)"
            }

        # --- 3. BIO-ACOUSTIC PHYSICS (The 'Macro' Features) ---
        # Note: These must match the training script scaling exactly.
        peak = np.max(np.abs(wav_data))
        safe_rms = rms + 1e-6
        
        crest = (peak / safe_rms) * 0.1  # Scaled Crest Factor (0-2 range usually)
        zcr = np.mean(np.abs(np.diff(np.sign(wav_data))) > 0) # Turbulence
        
        # ECR (Energy Concentration Ratio) - Measure the tightness of the burst
        mid = len(wav_data) // 2
        core = wav_data[max(0, mid-480):min(len(wav_data), mid+480)] # 60ms core window
        ecr = (np.sqrt(np.mean(core**2)) + 1e-6) / safe_rms

        # --- 4. DEEP LEARNING FEATURES (YAMNet) ---
        # scores_np: [frames, 521], embeddings_np: [frames, 1024]
        scores, embeddings, _ = self.yamnet(wav_data)
        scores_np = scores.numpy()
        embeddings_np = embeddings.numpy()
        
        # Average embeddings across the 5s segment
        avg_embedding = np.mean(embeddings_np, axis=0)
        
        # Find frame-level presence (Catches coughs hidden in speech)
        max_cough_presence = np.max(scores_np[:, 69]) 
        max_sneeze_presence = np.max(scores_np[:, 71])
        
        # Identify winning class and veto probabilities
        mean_scores = np.mean(scores_np, axis=0)
        winner_name = self.class_names[np.argmax(mean_scores)]
        veto_probs = mean_scores[self.veto_indices]

        # --- 5. FEATURE STACKING (The 1035 Vector) ---
        # [0:1024]      - YAMNet Embeddings
        # [1024:1027]   - Physics (Crest, ZCR, ECR)
        # [1027:1035]   - Veto Probabilities
        full_input = np.hstack([
            avg_embedding, 
            [crest, zcr, ecr], 
            veto_probs
        ]).reshape(1, 1035).astype(np.float32)

        self.interpreter.set_tensor(self.input_details, full_input)
        self.interpreter.invoke()
        
        prob_texture = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        prob_physics = self.interpreter.get_tensor(self.output_details[1]['index'])[0][0]
        
        # --- THE SPECIALIST FUSION ---
        # Instead of raw multiplication, we use a 'High-Confidence' fusion.
        # If both heads are even moderately sure, the score stays high.
        fused_score = (prob_texture + prob_physics) / 2.0
        
        # If one head is VERY sure it's NOISE (0.0), it should drag the other down.
        if prob_texture < 0.1 or prob_physics < 0.1:
            fused_score = prob_texture * prob_physics 
        # 4. HYBRID VERDICT
        # Even if the fused score is low, we still check the YAMNet "Cough Presence" 
        # as a safety bridge (Gate B from our previous discussion).
        # Use frame-level max to catch coughs hidden in speech
        max_cough_presence = np.max(scores_np[:, 69]) 
        max_sneeze_presence = np.max(scores_np[:, 71])
        
        # Check if the general YAMNet winner is clinical
        is_yamnet_clinical = any(x in winner_name for x in ["Cough", "Sneeze", "Respiratory"])
        peak_strength = (crest/0.1) * ecr # Scaled impact measure
        # --- 4. THE INTEGRATED DECISION TREE ---
        prediction = 0
        verdict_level = 0
        status = ""
        if fused_score < 0.01 and max_cough_presence < 0.05 and not is_yamnet_clinical:
            prediction, verdict_level = 0, 0
            status = "REJECTED (Brain Certainty)"
        # GATE A: THE EXPERT SURE-HIT (Using Fusion)
        elif fused_score > 0.70:
            prediction, verdict_level = 1, 1
            status = "ACCEPTED (Expert High Prob)"
        
        # GATE B: THE "PRESENCE" RESCUE (From your working old logic)
        # Catches coughs even if the whole segment is 'Speech'
        elif max_cough_presence > 0.10 or max_sneeze_presence > 0.10 or peak_strength > 4.0:
            prediction, verdict_level = 1, 2
            if max_cough_presence > 0.15:
                status = f"RESCUED (Cough Presence: {round(max_cough_presence,2)})"
            elif max_sneeze_presence > 0.10:
                status = f"RESCUED (Sneeze Presence: {round(max_sneeze_presence,2)})"
            else:
                status = f"RESCUED (Peak Presence: {round(peak_strength,2)} ecr {ecr} crest {crest})"
        # GATE C: THE EXPERT SUSPECT (Using Fusion)
        elif fused_score > 0.15:
            prediction, verdict_level = 1, 2
            status = "SUSPECT (Expert Mid Prob)"
        
        # GATE D: THE STANDARD AI RESCUE (From your working old logic)
        elif is_yamnet_clinical:
            prediction, verdict_level = 1, 2
            status = f"RESCUED (YAMNet {winner_name} Hint)"
            
        else:
            prediction, verdict_level = 0, 0
            status = "REJECTED"

        # Updated Debug Line to show all components
        # print(f"DEBUG: [{segment_tag[:18]:<18}] Win: {winner_name:<12} | Fused: {fused_score:.2f} | MaxPulse: {max_cough_presence:.2f} | {status}")
        return {
    "preds": int(prediction), 
    "probs": float(fused_score), 
    "verdict_level": int(verdict_level), 
    "top_class": winner_name}