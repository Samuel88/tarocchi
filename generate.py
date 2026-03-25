#!/usr/bin/env python3
"""
Genera le 22 immagini dei tarocchi in batch usando Replicate API.

Requisiti:
    pip install replicate requests

Setup:
    export REPLICATE_API_TOKEN="r8_il_tuo_token_qui"
    (Ottieni il token gratis su https://replicate.com/account/api-tokens)

Uso:
    python generate.py
"""

import os
import re
import sys
import time
import requests
import replicate


PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Modello Animagine XL 3.1 su Replicate
MODEL = "cjwbw/animagine-xl-3.1:6afe2e6b1f68e4e39caf0cbc1ded1a20fcde25e835fb93c9e066e8046c0702e7"

# Parametri di generazione
WIDTH = 640
HEIGHT = 896
NUM_STEPS = 35
GUIDANCE_SCALE = 7.5
SCHEDULER = "DPM++ 2M Karras"


def parse_prompts(filepath):
    """Estrae i prompt e il negative prompt dal file prompts.txt."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Estrai tutti i blocchi tra triple backtick
    blocks = re.findall(r"```\n(.*?)```", content, re.DOTALL)
    if not blocks:
        print("Errore: nessun prompt trovato nel file.")
        sys.exit(1)

    # Il primo blocco è il negative prompt
    negative_prompt = blocks[0].strip()

    # I restanti sono i prompt delle carte
    card_prompts = []
    for block in blocks[1:]:
        card_prompts.append(block.strip())

    # Estrai i nomi delle carte dai titoli ##
    card_names = re.findall(r"## (.+?)$", content, re.MULTILINE)
    # Rimuovi il primo titolo (Negative prompt)
    card_names = [n for n in card_names if "Negative" not in n]

    return negative_prompt, card_prompts, card_names


def sanitize_filename(name):
    """Converte il nome della carta in un nome file sicuro."""
    # Rimuovi numeri romani e trattini iniziali
    name = re.sub(r"^[IVXLCDM0-9]+\s*[—–-]\s*", "", name)
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def generate_image(prompt, negative_prompt, card_index, card_name):
    """Genera una singola immagine via Replicate API."""
    output = replicate.run(
        MODEL,
        input={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": WIDTH,
            "height": HEIGHT,
            "num_inference_steps": NUM_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "scheduler": SCHEDULER,
            "num_outputs": 1,
        },
    )

    # L'output è una lista di URL
    if not output:
        print(f"  Errore: nessun output per {card_name}")
        return None

    image_url = output[0] if isinstance(output, list) else output

    # Se è un FileOutput, convertilo a stringa
    image_url = str(image_url)

    # Scarica l'immagine
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()

    safe_name = sanitize_filename(card_name)
    filename = f"generated_{card_index:02d}_{safe_name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


def main():
    # Verifica token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Errore: imposta la variabile REPLICATE_API_TOKEN")
        print("  export REPLICATE_API_TOKEN='r8_il_tuo_token'")
        print("  (Ottieni il token su https://replicate.com/account/api-tokens)")
        sys.exit(1)

    print("Parsing dei prompt...")
    negative_prompt, card_prompts, card_names = parse_prompts(PROMPTS_FILE)

    print(f"Trovati {len(card_prompts)} prompt")
    print(f"Negative prompt: {negative_prompt[:80]}...")
    print(f"Risoluzione: {WIDTH}x{HEIGHT}")
    print(f"Modello: Animagine XL 3.1")
    print("-" * 50)

    successes = 0
    failures = 0

    for i, (prompt, name) in enumerate(zip(card_prompts, card_names)):
        print(f"\n[{i + 1}/{len(card_prompts)}] Generazione: {name}")
        print(f"  Prompt: {prompt[:70]}...")

        try:
            filepath = generate_image(prompt, negative_prompt, i, name)
            if filepath:
                print(f"  Salvata: {os.path.basename(filepath)}")
                successes += 1
            else:
                failures += 1
        except Exception as e:
            print(f"  Errore: {e}")
            failures += 1
            # Pausa breve in caso di rate limiting
            time.sleep(2)
            continue

        # Piccola pausa tra le richieste per evitare rate limiting
        time.sleep(1)

    print("\n" + "=" * 50)
    print(f"Completato! {successes} immagini generate, {failures} errori.")
    print(f"Immagini salvate in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
