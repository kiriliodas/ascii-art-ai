import os
import torch
import streamlit as st
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# === Constantes ===
MODEL_DIR = "./ascii_art_model"

# === Fonctions ===
def format_conversations(example):
    """Formate les données en texte utilisable pour l'entraînement."""
    user_content = example['conversations'][0]['content']
    assistant_content = example['conversations'][1]['content']
    formatted_text = f"User: {user_content}\nAssistant: {assistant_content}\n"
    return {"text": formatted_text}

def tokenize_function(example, tokenizer):
    """Tokenise les exemples pour GPT-2."""
    return tokenizer(example["text"], truncation=True, max_length=512)

# === Interface Streamlit ===
st.title("🎨 IA Générateur d'Art ASCII")

# === Chargement du modèle existant ===
if os.path.exists(MODEL_DIR):
    st.success("✅ Modèle chargé depuis le disque !")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
else:
    st.warning("⚠️ Aucun modèle trouvé. Entraîne-le d'abord pour pouvoir générer !")
    tokenizer = None
    model = None

# === Section 1 : Génération d’art ASCII ===
st.header("🖋️ Générer un Art ASCII")

prompt = st.text_input("Entre ton prompt (ex: 'Draw a cat in ASCII art')")
max_length = st.slider("Longueur max de génération", 100, 1024, 512)

if st.button("Générer"):
    if model:
        with st.spinner("Génération en cours..."):
            input_text = f"User: {prompt}\nAssistant:"
            inputs = tokenizer.encode(input_text, return_tensors="pt")

            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_p=0.95
            )

            generated_text = tokenizer.decode(outputs[0])
            assistant_response = (
                generated_text.split("Assistant:")[1].strip()
                if "Assistant:" in generated_text else generated_text
            )

            st.code(assistant_response, language="text")  # Garde le format ASCII
    else:
        st.error("🚫 Entraîne le modèle d'abord avant de générer.")

# === Section 2 : Entraînement du modèle ===
st.header("⚙️ Entraîner / Améliorer le Modèle")
st.info("💡 L'entraînement peut être long. Utilise un GPU si possible. Limite le dataset pour tester.")

epochs = st.number_input("Nombre d'époques", min_value=1, max_value=10, value=3)
batch_size = st.number_input("Batch size par device", min_value=1, max_value=16, value=4)
dataset_size = st.number_input("Taille du dataset (0 = tout utiliser)", min_value=0, value=0)

if st.button("Lancer l'Entraînement"):
    with st.spinner("Chargement du dataset et entraînement en cours..."):
        # Chargement du dataset
        dataset = load_dataset("mrzjy/ascii_art_generation_140k", split="train")
        if dataset_size > 0:
            dataset = dataset.shuffle(seed=42).select(range(dataset_size))

        # Préparation du tokenizer et du modèle
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if tokenizer is None else tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        formatted_dataset = dataset.map(format_conversations, remove_columns=dataset.column_names)
        tokenized_dataset = formatted_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer),
            batched=True,
            remove_columns=["text"]
        )

        model = (
            GPT2LMHeadModel.from_pretrained(MODEL_DIR)
            if os.path.exists(MODEL_DIR)
            else GPT2LMHeadModel.from_pretrained("gpt2")
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            overwrite_output_dir=False,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=5e-5,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        # Sauvegarde du modèle
        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        # Enregistrement du log d'entraînement
        with open("training_log.txt", "a") as f:
            loss = trainer.state.log_history[-1].get('train_loss', 'N/A') if trainer.state.log_history else 'N/A'
            f.write(f"Époque {epochs} terminée. Loss finale : {loss}\n")

        st.success("✅ Entraînement terminé et modèle sauvegardé ! Recharge la page pour le recharger.")

# === Section 3 : Logs d'entraînement ===
if st.button("📜 Voir l'évolution des entraînements"):
    if os.path.exists("training_log.txt"):
        with open("training_log.txt", "r") as f:
            st.text(f.read())
    else:
        st.info("Aucun entraînement lancé pour le moment.")

# === Pied de page ===
st.markdown("---")
st.text("🧠 Fait avec Streamlit — déploie sur un serveur pour l’utiliser à distance.")
