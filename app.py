from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms

app = FastAPI()

path_to_labels = "path"
path_to_model = "path"
checkpoint = torch.load(path_to_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model = checkpoint['ema']

model = model.to(torch.float32)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def labels_reading(path_to_labels):
    with open(path_to_labels, 'r', encoding='utf-8') as file:
        class_names = [line.strip() for line in file]
    return class_names


def predict(image_tensor):
    with torch.no_grad():
        results = model(image_tensor)
    class_num = results.argmax(dim=1).item()
    class_names = labels_reading(path_to_labels)
    return class_names[class_num]


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    img_tensor = transform(image).unsqueeze(0).to(torch.float32)  # Приведение входных данных к формату torch.float32
    predicted_class = predict(img_tensor)
    return JSONResponse(content={"class": predicted_class})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)