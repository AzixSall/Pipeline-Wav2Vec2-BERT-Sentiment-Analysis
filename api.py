from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipeline import Wav2VecBERTPipeline
import tempfile
import os

app = FastAPI(title="Wav2Vec2 + BERT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Wav2VecBERTPipeline()

@app.get("/")
async def root():
    return {"message": "Wav2Vec2 + BERT API", "status": "active"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), use_chunks: bool = False):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Format non supporté")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        if use_chunks:
            result = pipeline.process_long_audio(tmp_path)
        else:
            result = pipeline.process_audio(tmp_path)
        
        os.unlink(tmp_path)
        return result
        
    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)