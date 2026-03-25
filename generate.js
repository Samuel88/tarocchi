#!/usr/bin/env node
/**
 * Genera le 22 immagini dei tarocchi in batch usando Replicate API (Flux Dev).
 *
 * Requisiti:
 *     npm install replicate
 *
 * Setup:
 *     export REPLICATE_API_TOKEN="r8_il_tuo_token_qui"
 *     (Ottieni il token su https://replicate.com/account/api-tokens)
 *
 * Uso:
 *     node generate.js
 */

import Replicate from "replicate";
import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join, basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PROMPTS_FILE = join(__dirname, "prompts.json");
const OUTPUT_DIR = __dirname;

// Modello Flux Dev su Replicate
const MODEL = "black-forest-labs/flux-dev";

// Parametri di generazione
const ASPECT_RATIO = "3:4";
const MEGAPIXELS = "1";
const NUM_STEPS = 28;
const GUIDANCE = 3.5;
const OUTPUT_FORMAT = "png";
const OUTPUT_QUALITY = 100;

function loadPrompts(filepath) {
  const data = JSON.parse(readFileSync(filepath, "utf-8"));
  if (!data.cards || data.cards.length === 0) {
    console.error("Errore: nessuna carta trovata nel file JSON.");
    process.exit(1);
  }
  return data;
}

async function generateImage(replicate, prompt, cardIndex, filename) {
  const [image] = await replicate.run(MODEL, {
    input: {
      prompt,
      go_fast: false,
      guidance: GUIDANCE,
      megapixels: MEGAPIXELS,
      num_outputs: 1,
      aspect_ratio: ASPECT_RATIO,
      output_format: OUTPUT_FORMAT,
      output_quality: OUTPUT_QUALITY,
      prompt_strength: 0.8,
      num_inference_steps: NUM_STEPS,
    },
  });

  if (!image) {
    console.log(`  Errore: nessun output per ${filename}`);
    return null;
  }

  const outputFilename = `generated_${String(cardIndex).padStart(2, "0")}_${filename}.png`;
  const filepath = join(OUTPUT_DIR, outputFilename);

  // FileObject è un ReadableStream, writeFile lo gestisce direttamente
  await writeFile(filepath, image);

  return filepath;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  if (!process.env.REPLICATE_API_TOKEN) {
    console.error("Errore: imposta la variabile REPLICATE_API_TOKEN");
    console.error("  export REPLICATE_API_TOKEN='r8_il_tuo_token'");
    console.error("  (Ottieni il token su https://replicate.com/account/api-tokens)");
    process.exit(1);
  }

  const replicate = new Replicate();

  console.log("Caricamento prompt...");
  const { cards } = loadPrompts(PROMPTS_FILE);

  console.log(`Trovati ${cards.length} prompt`);
  console.log(`Aspect ratio: ${ASPECT_RATIO}`);
  console.log(`Modello: Flux Dev`);
  console.log("-".repeat(50));

  let successes = 0;
  let failures = 0;

  for (let i = 0; i < 2; i++) {
    const { name, filename, prompt } = cards[i];

    console.log(`\n[${i + 1}/${cards.length}] Generazione: ${name}`);
    console.log(`  Prompt: ${prompt.slice(0, 70)}...`);

    try {
      const filepath = await generateImage(replicate, prompt, i, filename);
      if (filepath) {
        console.log(`  Salvata: ${basename(filepath)}`);
        successes++;
      } else {
        failures++;
      }
    } catch (e) {
      console.error(`  Errore: ${e.message}`);
      failures++;
      await sleep(2000);
      continue;
    }

    await sleep(15 * 1000);
  }

  console.log("\n" + "=".repeat(50));
  console.log(`Completato! ${successes} immagini generate, ${failures} errori`);
  console.log(`Immagini salvate in: ${OUTPUT_DIR}`);
}

main();
