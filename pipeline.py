import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import re

class Wav2VecBERTPipeline:
    def __init__(self, 
                 wav2vec_model="facebook/wav2vec2-large-960h-lv60-self",
                 bert_model="nlptown/bert-base-multilingual-uncased-sentiment"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model)
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model)
        self.wav2vec_model.to(self.device)
        self.wav2vec_model.eval()
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=bert_model,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True
        )
        
        self.label_mapping = {
            "1 star": "très négatif",
            "2 stars": "négatif", 
            "3 stars": "neutre",
            "4 stars": "positif",
            "5 stars": "très positif"
        }
    
    def transcribe(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = librosa.util.normalize(audio)
        
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.wav2vec_model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        confidence = torch.mean(max_probs).item()
        
        return {
            "text": transcription.strip(),
            "confidence": confidence
        }
    
    def analyze_sentiment(self, text):
        if not text or len(text.strip()) < 3:
            return {"sentiment": "neutre", "confidence": 0.0, "scores": {}}
        
        text = re.sub(r'[^\w\s\.,!?àáâäçéèêëíìîïñóòôöúùûüýÿ]', '', text)
        text = ' '.join(text.split())
        
        words = text.split()
        if len(words) > 100:
            text = ' '.join(words[:100])
        
        results = self.sentiment_pipeline(text)
        
        scores = {}
        max_score = 0
        predicted_sentiment = "neutre"
        
        for result in results[0]:
            label = result["label"]
            score = result["score"]
            mapped_label = self.label_mapping.get(label, label.lower())
            scores[mapped_label] = score
            
            if score > max_score:
                max_score = score
                predicted_sentiment = mapped_label
        
        simplified_scores = {"négatif": 0.0, "neutre": 0.0, "positif": 0.0}
        
        for label, score in scores.items():
            if "négatif" in label:
                simplified_scores["négatif"] += score
            elif "positif" in label:
                simplified_scores["positif"] += score
            else:
                simplified_scores["neutre"] += score
        
        simplified_sentiment = "neutre"
        if "négatif" in predicted_sentiment:
            simplified_sentiment = "négatif"
        elif "positif" in predicted_sentiment:
            simplified_sentiment = "positif"
        
        return {
            "sentiment": simplified_sentiment,
            "confidence": max_score,
            "scores": simplified_scores
        }
    
    def process_audio(self, audio_path):
        transcription = self.transcribe(audio_path)
        sentiment = self.analyze_sentiment(transcription["text"])
        
        return {
            "text": transcription["text"],
            "transcription_confidence": transcription["confidence"],
            "sentiment": sentiment["sentiment"],
            "sentiment_confidence": sentiment["confidence"],
            "sentiment_scores": sentiment["scores"]
        }
    
    def process_long_audio(self, audio_path, chunk_duration=30):
        audio, sr = librosa.load(audio_path, sr=16000)
        chunk_samples = chunk_duration * sr
        
        results = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > sr:
                import tempfile
                import soundfile as sf
                import os
                
                tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp_path = tmp_file.name
                tmp_file.close()

                try:
                    sf.write(tmp_path, chunk, sr)
                    result = self.process_audio(tmp_path)
                    results.append(result)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        full_text = " ".join([r["text"] for r in results if r["text"]])
        
        sentiments = [r["sentiment"] for r in results]
        sentiment_counts = {"négatif": 0, "neutre": 0, "positif": 0}
        for s in sentiments:
            sentiment_counts[s] += 1
        
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiments else "neutre"
        
        confidences = [r["sentiment_confidence"] for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "full_text": full_text,
            "overall_sentiment": overall_sentiment,
            "confidence": avg_confidence,
            "segments": results,
            "summary": {
                "total_segments": len(results),
                "distribution": {k: {"count": v, "percentage": round((v/len(results))*100, 1) if results else 0}
                               for k, v in sentiment_counts.items()}
            }
        }