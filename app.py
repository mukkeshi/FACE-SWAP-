import os
import torch
import cv2
from PIL import Image
from flask import Flask, render_template, request
from transformers import AutoModelForImageClassification, AutoImageProcessor

app = Flask(__name__)

# மாடல் லோட் செய்தல் (Render-ல் லோக்கல் பாதை அல்லது HuggingFace பயன்படுத்தலாம்)
MODEL_PATH = "final_swin_deepfake_model" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # static ஃபோல்டர் இல்லை என்றால் உருவாக்குவதற்கு
            os.makedirs('static', exist_ok=True)
            
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Face Detection & Crop
            img_cv = cv2.imread(filepath)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cropped = img_cv[y:y+h, x:x+w]
                raw_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            else:
                raw_image = Image.open(filepath).convert("RGB")

            # Prediction
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            fake_score = round(probs[0].item() * 100, 2)
            real_score = round(probs[1].item() * 100, 2)

            result = "REAL ✅" if real_score > fake_score else "FAKE ❌"
            confidence = max(real_score, fake_score)

            return render_template('index.html', result=result, confidence=confidence, image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    # Render Dynamic Port Binding
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
