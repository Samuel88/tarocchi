#!/usr/bin/env python3
"""
Genera le 22 immagini dei tarocchi in batch usando Replicate API (Flux Dev).

Requisiti:
    pip install replicate

Setup:
    export REPLICATE_API_TOKEN="r8_il_tuo_token_qui"
    (Ottieni il token su https://replicate.com/account/api-tokens)

Uso:
    python generate.py
"""

import os
import re
import sys
import time

import replicate

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Modello Flux Dev su Replicate
MODEL = "black-forest-labs/flux-dev"

# Parametri di generazione
ASPECT_RATIO = "5:7"
MEGAPIXELS = "1"
NUM_STEPS = 28
GUIDANCE = 3.5
OUTPUT_FORMAT = "png"
OUTPUT_QUALITY = 100


def parse_prompts(filepath):
    """Estrae i prompt dal file prompts.txt."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Estrai tutti i blocchi tra triple backtick
    blocks = re.findall(r"```\n(.*?)```", content, re.DOTALL)
    if not blocks:
        print("Errore: nessun prompt trovato nel file.")
        sys.exit(1)

    # Il primo blocco è il negative prompt (Flux non lo supporta,
    # ma lo integriamo nel prompt come indicazione)
    negative_keywords = blocks[0].strip()

    # I restanti sono i prompt delle carte
    card_prompts = []
    for block in blocks[1:]:
        card_prompts.append(block.strip())

    # Estrai i nomi delle carte dai titoli ##
    card_names = re.findall(r"## (.+?)$", content, re.MULTILINE)
    card_names = [n for n in card_names if "Negative" not in n]

    return negative_keywords, card_prompts, card_names


def sanitize_filename(name):
    """Converte il nome della carta in un nome file sicuro."""
    name = re.sub(r"^[IVXLCDM0-9]+\s*[—–-]\s*", "", name)
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name



def generate_image(prompt, card_index, card_name):
    """Genera una singola immagine via Replicate API con Flux Dev."""
    output = replicate.run(
        MODEL,
        input={
            "prompt": prompt,
            "go_fast": True,
            "guidance": GUIDANCE,
            "megapixels": MEGAPIXELS,
            "num_outputs": 1,
            "aspect_ratio": ASPECT_RATIO,
            "output_format": OUTPUT_FORMAT,
            "output_quality": OUTPUT_QUALITY,
            "prompt_strength": 0.8,
            "num_inference_steps": NUM_STEPS,
        },
    )

    if not output:
        print(f"  Errore: nessun output per {card_name}")
        return None

    # Salva l'immagine usando .read() come da API Flux
    safe_name = sanitize_filename(card_name)
    filename = f"generated_{card_index:02d}_{safe_name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(output[0].read())

    return filepath


def main():
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Errore: imposta la variabile REPLICATE_API_TOKEN")
        print("  export REPLICATE_API_TOKEN='r8_il_tuo_token'")
        print("  (Ottieni il token su https://replicate.com/account/api-tokens)")
        sys.exit(1)

    print("Parsing dei prompt...")
    negative_keywords, card_prompts, card_names = parse_prompts(PROMPTS_FILE)

    print(f"Trovati {len(card_prompts)} prompt")
    print(f"Aspect ratio: {ASPECT_RATIO}")
    print(f"Modello: Flux Dev")
    print("-" * 50)

    successes = 0
    failures = 0

    for i, (prompt, name) in enumerate(zip(card_prompts, card_names)):
        print(f"\n[{i + 1}/{len(card_prompts)}] Generazione: {name}")
        print(f"  Prompt: {prompt[:70]}...")

        try:
            filepath = generate_image(prompt, i, name)
            if filepath:
                print(f"  Salvata: {os.path.basename(filepath)}")
                successes += 1
            else:
                failures += 1
        except Exception as e:
            print(f"  Errore: {e}")
            failures += 1
            time.sleep(2)
            continue

        time.sleep(1)

    print("\n" + "=" * 50)
    print(f"Completato! {successes} immagini generate, {failures} errori.")
    print(f"Immagini salvate in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
