import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

"""Script que se asegura de que el modelo y las transformaciones se carguen una sola vez de forma correcta."""

NUM_CLASSES = 7
DEVICE = torch.device("cpu") # Se usa CPU
MODEL_PATH = '../model/best_skin_cancer_model.pth'
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def get_model_transforms():
    """Define las transformaciones exactamente igual que el entrenamiento."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_model():
    """Carga el modelo pre-entrenado y los pesos guardados."""
    try:
        # Cargar ResNet-18 sin pesos pre-entrenados, 
        model = models.resnet18(weights=None) 
        
        # Modificar la capa final (fc)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
        # Cargar los pesos entrenados
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Modo evaluación
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo o los pesos: {e}")
        return None


def predict_image(image_bytes: bytes, model, transform):
    """
    Función de inferencia principal. Toma bytes de imagen, los procesa y ejecuta la predicción.
    """
    # Leer los bytes como una imagen PIL
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Aplicar transformaciones
    image_tensor = transform(image)
    
    # Añadir (BATCH) -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Ejecutar el forward pass
        output = model(image_tensor)
        
        # Calcular la probabilidad (softmax)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Obtener la clase predicha y su confianza
        confidence, predicted_index = torch.max(probabilities, 0)
    
    # Map de índice numérico a nombre de clase
    predicted_class = CLASS_NAMES[predicted_index.item()]
    
    return predicted_class, confidence.item()

# Inicializar 
model = load_model()
transform = get_model_transforms()