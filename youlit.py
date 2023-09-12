import streamlit as lit
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model import LlamaCPPInvocationLayer
import time

lit.set_page_config(
    layout="wide"
)

def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():

    # Set the title and background color
    lit.title("YouLitAI: AI Youtube Video Summarizer")
    lit.markdown('<style>h1{color: #45f383; text-align: center;}</style>', unsafe_allow_html=True)
    lit.subheader('Built with ❤️ By Khalifa MBA')
    lit.markdown('<style>h3{color: #51ffe4;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with lit.expander("About YouLitAI"):
        lit.write("YouLitAI summarizes YouTube videos with AI.")
        lit.write("To start, enter a YouTube Video URL in the box below and click 'Summarize'.")

    # Input box for YouTube URL
    youtube_url = lit.text_input("Enter YouTube Video URL")

    # Summarize button
    if lit.button("Summarize") and youtube_url:
        start_time = time.time()  # Start the timer
        # Download video
        file_path = download_video(youtube_url)

        # Initialize model
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = prompt_node = initialize_prompt_node(model)
        # Transcribe audio
        output = transcribe_audio(file_path, prompt_node)

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = lit.columns([1,1])

        # Column 1: Video view
        with col1:
            lit.video(youtube_url)

        # Column 2: Summary View
        with col2:
            lit.header("YouLitAI: AI Youtube Video Summarizer")
            lit.write(output)
            lit.success(output["results"][0].split("\n\n[INST]")[0])
            lit.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
