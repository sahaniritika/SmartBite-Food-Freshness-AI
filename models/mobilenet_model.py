import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Preprocessing Pipeline (supports RGB conversion)
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

def predict_spoilage(image_pil, model):
    # RGBA (PNG 4-channel) ko 3-channel RGB me convert karna
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
        
    input_tensor = transform_pipeline(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    spoiled_score = float(probabilities[0]) 
    return round(spoiled_score * 100, 2)