import torch
import argparse
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex import NougatLaTexProcessor

# --- Configuration ---
model_name = "Norm/nougat-latex-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_image_to_latex(image_path):
    """
    Loads an image, processes it with the Nougat-LaTex model, and returns
    the generated LaTeX string.
    """
    print(f"Loading model and tokenizer from {model_name}...")
    
    # Init model
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    # Init tokenizer
    tokenizer = NougatTokenizerFast.from_pretrained(model_name)
    # Init processor
    latex_processor = NougatLaTexProcessor.from_pretrained(model_name)

    print(f"Processing image: {image_path} on device: {device}...")
    
    # Load and prepare image
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
        
    if not image.mode == "RGB":
        image = image.convert('RGB')

    # Preprocess pixel values
    pixel_values = latex_processor(image, return_tensors="pt").pixel_values

    # Prepare decoder input (BOS token)
    decoder_input_ids = tokenizer(tokenizer.bos_token, add_special_tokens=False,
                                  return_tensors="pt").input_ids
    
    # --- Generate LaTeX Output ---
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=5,
            bad_words_ids=[[tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    # Decode and clean sequence
    sequence = tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
    
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image of LaTeX to its text representation using Nougat-LaTex.")
    parser.add_argument("image_path", type=str, help="Path to the input image (e.g., 'path/to/formula.png')")
    args = parser.parse_args()
    
    latex_output = process_image_to_latex(args.image_path)
    
    if latex_output:
        print("\n--- Final LaTeX Output ---")
        print(latex_output)
        print("--------------------------")
