import gradio as gr
from pipeline import Wav2VecBERTPipeline

pipeline = Wav2VecBERTPipeline()

def analyze_audio(audio_file, processing_mode):
    if not audio_file:
        return "Uploadez un fichier audio"
    
    yield "Chargement des modèles..."

    try:
        if processing_mode == "Court (< 1 min)":
            yield "Transcription en cours..."
            result = pipeline.process_audio(audio_file)
            yield "Analyse du sentiment..."
            output = f"""
## Transcription
{result['text']}

## Sentiment
**Résultat:** {result['sentiment'].upper()}
**Confiance:** {result['sentiment_confidence']:.1%}

### Scores:
"""
            for sentiment, score in result['sentiment_scores'].items():
                output += f"- {sentiment.capitalize()}: {score:.1%}\n"
                
        else:
            yield "Transcription par chunks..."
            result = pipeline.process_long_audio(audio_file)
            yield "Analyse globale..."
            output = f"""
## Transcription complète
{result['full_text'][:300]}{'...' if len(result['full_text']) > 300 else ''}

## Sentiment global
**Résultat:** {result['overall_sentiment'].upper()}
**Confiance:** {result['confidence']:.1%}

### Distribution:
"""
            for sentiment, data in result['summary']['distribution'].items():
                if data['count'] > 0:
                    output += f"- {sentiment.capitalize()}: {data['count']} segments ({data['percentage']:.1f}%)\n"
        
        yield output
        
    except Exception as e:
        return f"Erreur: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Wav2Vec2 + BERT Pipeline")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Fichier Audio", type="filepath")
            processing_mode = gr.Radio(
                choices=["Court (< 1 min)", "Long (chunks)"],
                value="Court (< 1 min)",
                label="Mode"
            )
            analyze_btn = gr.Button("Analyser", variant="primary")
        
        with gr.Column():
            output = gr.Markdown()
    
    analyze_btn.click(analyze_audio, inputs=[audio_input, processing_mode], outputs=output, show_progress=True)
    
    gr.Markdown("**Formats:** WAV, MP3, FLAC")

if __name__ == "__main__":
    demo.launch()