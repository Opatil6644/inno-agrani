"""
This script serves as a Text-to-Speech (TTS) profiling and generation router. 
It allows users to generate high-quality audio from text using two distinct 
"""


import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import argparse
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# function to run inference for ai4bharat model
def ai4_bharat(text, output_path):
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts", revision="7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca").to(device)
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", revision="7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca")
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        prompt = text
        description = "A female speaker with a American accent delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

        description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

        generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(f"{output_path}/tts_output.wav", audio_arr, model.config.sampling_rate)
    except Exception as e:
        print(f"Error occurred: {e}")


# function to run inference for parler-tts model
def parler_tts(text, output_path):
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1", revision="50cb4b874c83902f930d7c2e753224c15654f11e").to(device)
        tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1", revision="50cb4b874c83902f930d7c2e753224c15654f11e")

        prompt = text
        description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(f"{output_path}/tts_output.wav", audio_arr, model.config.sampling_rate)

    except Exception as e:
        print(f"Error occurred: {e}")


# --- MAIN ROUTER ---
def main():
    parser = argparse.ArgumentParser(description="TTS Performance Profiling Router")
    parser.add_argument("--model_name", required=True, help="Model to profile")
    parser.add_argument("--text", type=str, default="Testing the performance of this text to speech model.", help="Text to synthesize")
    parser.add_argument("--results_path", help="Output path to save the audio")
    
    args = parser.parse_args()

    if args.model_name.startswith("ai4bharat"):
        ai4_bharat(args.text, args.results_path)
    
    elif args.model_name.startswith("parler-tts"):
        parler_tts(args.text, args.results_path)

    else:
        print(f"Invalid Model name: {args.model_name}")

if __name__ == "__main__":
    main()