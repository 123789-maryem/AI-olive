from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "olive_disease_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully")
print("Input details:", input_details)
print("Output details:", output_details)



CLASS_NAMES = [
    "Aculus_Olearius",
    "Anthracnose",
    "Black_Scale",
    "Cycloconium",
    "Fumagina",
    "Healthy",
    "Knot_Disease",
    "Nutritional_Deficiency",
    "Peacock_Spot",
    "Rust_Mite",
    "Virosis"
]


DISEASE_INFO = {

    "Aculus_Olearius": {
        "description": "Acarien microscopique (Aculus olearius) qui provoque une déformation, un jaunissement et un dessèchement des jeunes feuilles et pousses de l'olivier.",
        "treatment": "Appliquer un acaricide spécifique au printemps. Tailler les parties infestées et améliorer l'aération de l'arbre. Éviter l'excès d'azote."
    },

    "Anthracnose": {
        "description": "Maladie fongique (Colletotrichum) provoquant des taches brunes sur les fruits et les feuilles, entraînant leur pourriture, surtout par temps humide.",
        "treatment": "Traitement à base de cuivre (bouillie bordelaise) avant les pluies automnales. Éliminer les fruits et feuilles infectés. Assurer une bonne aération de la frondaison."
    },

    "Black_Scale": {
        "description": "Infestation par la cochenille noire (Saissetia oleae), un insecte qui suce la sève et produit du miellat, favorisant le développement de fumagine.",
        "treatment": "Traitement à l'huile blanche ou insecticide spécifique. Introduire des prédateurs naturels (coccinelles). Tailler pour aérer l'arbre."
    },

    "Cycloconium": {
        "description": "Maladie fongique (Cycloconium oleaginum), aussi appelée œil de paon, causant des taches circulaires brun-noir sur les feuilles.",
        "treatment": "Traitement cuprique préventif en automne et au printemps. Ramasser et détruire les feuilles tombées. Éviter les excès d'humidité."
    },

    "Fumagina": {
        "description": "Champignon noir (fumagine) se développant sur le miellat sécrété par des insectes piqueurs-suceurs (cochenilles, pucerons), recouvrant les feuilles d'une couche noire.",
        "treatment": "Traiter en premier lieu les insectes responsables (cochenilles). Nettoyer les feuilles à l'eau savonneuse. Appliquer un fongicide si nécessaire."
    },

    "Healthy": {
        "description": "La feuille analysée ne présente aucun signe visible de maladie ou de parasite. L'arbre semble en bonne santé.",
        "treatment": "Aucun traitement nécessaire. Continuer un entretien régulier : arrosage adapté, taille annuelle et surveillance périodique."
    },

    "Knot_Disease": {
        "description": "Maladie bactérienne (tuberculose de l'olivier, Pseudomonas savastanoi) provoquant des excroissances (nodosités) sur les branches et le tronc.",
        "treatment": "Tailler et brûler les branches atteintes (désinfecter les outils entre chaque coupe). Appliquer un traitement cuprique après la taille. Éviter les blessures sur l'arbre."
    },

    "Nutritional_Deficiency": {
        "description": "Symptômes de carence nutritive (souvent en azote, fer ou magnésium) se traduisant par un jaunissement des feuilles ou une croissance ralentie.",
        "treatment": "Effectuer une analyse du sol pour identifier l'élément manquant. Apporter un engrais équilibré adapté à l'olivier. Corriger le pH du sol si nécessaire."
    },

    "Peacock_Spot": {
        "description": "Maladie fongique très répandue (Spilocaea oleagina), provoquant des taches circulaires brunes entourées d'un halo jaune sur les feuilles, causant leur chute.",
        "treatment": "Traitement cuprique en automne et fin d'hiver. Ramasser les feuilles tombées au sol. Tailler pour améliorer la circulation de l'air."
    },

    "Rust_Mite": {
        "description": "Acarien microscopique provoquant une décoloration bronze/rouille des feuilles et fruits, ainsi qu'un dessèchement progressif.",
        "treatment": "Application d'un acaricide adapté. Améliorer l'irrigation en période de stress hydrique. Surveiller régulièrement les jeunes pousses."
    },

    "Virosis": {
        "description": "Infection virale affectant l'olivier, provoquant des déformations, mosaïques ou décolorations sur les feuilles, sans traitement curatif direct.",
        "treatment": "Aucun traitement curatif. Éliminer les arbres fortement atteints pour éviter la propagation. Utiliser du matériel de greffe certifié sain."
    }
}



UPLOAD_FOLDER = "uploads"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)


def load_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def save_history(entry):
    history = load_history()
    history.insert(0, entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)



@app.route("/")
def home():
    return "Olive Disease AI is running!"



@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image provided"
        }), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({
            "error": "Invalid image"
        }), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    img.save(filepath)

    
    img_resized = img.resize((224, 224))


    img_array = np.array(img_resized).astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)


    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    interpreter.set_tensor(
        input_index,
        img_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_index
    )

    # Get prediction
    probabilities = prediction[0]

    idx = int(np.argmax(probabilities))

    disease = CLASS_NAMES[idx]

    confidence = float(probabilities[idx])

    # =========================
    # Disease information
    # =========================

    info = DISEASE_INFO.get(
        disease,
        {
            "description": "Aucune information disponible pour cette classe.",
            "treatment": "Consultez un expert agricole."
        }
    )

    # =========================
    # History
    # =========================

    entry = {
        "id": filename,
        "image_url": f"/uploads/{filename}",
        "disease": disease,
        "confidence": confidence,
        "description": info["description"],
        "treatment": info["treatment"],
        "date": datetime.now().isoformat()
    }

    save_history(entry)

    return jsonify(entry)


@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )