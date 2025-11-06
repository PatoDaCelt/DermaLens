from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import model_loader # Script con la lógica del modelo
import io

"""Script para crear el endpoint web"""

app = FastAPI(title="API de modelo de DermaLens")

if model_loader.model is None:
    raise RuntimeError("La API no puede iniciarse: el modelo no se cargó.")

@app.get("/", summary="Estado de la API")
async def root():
    """Health Endpoint para verificar que la API está funcionando."""
    return {"message": "API de DermaLens activa", "model_status": "OK"}

@app.post("/predict", summary="Clasificar una imagen de lesión en la piel")
async def classify_lesion(file: UploadFile = File(...)):
    """
    Recibe una imagen (JPEG/PNG/JPG) y devuelve la label de la lesión.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Tipo de archivo no soportado. Por favor, sube una imagen JPG, JPEG o PNG."
        )

    try:
        #Leer la imagen como bytes
        image_bytes = await file.read()
        
        #Ejecutar la inferencia
        predicted_class, confidence = model_loader.predict_image(
            image_bytes, 
            model_loader.model, 
            model_loader.transform
        )

        #Devolver el output en JSON
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4) #4 decimales
        })

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor al procesar la imagen."
        )