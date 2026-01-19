from src.inference.predict import load_model, predict_currency

model = load_model("best_currency_model.pth")
for i in range (1,5):
    image=f"image{i}.jpeg"
    result = predict_currency(image, model)
    print(f"image{i}:{result}")
