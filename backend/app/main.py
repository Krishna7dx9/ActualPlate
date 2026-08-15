from fastapi import FastAPI
from backend.routes.nutrition import router as nutrition_router
from backend.routes.upload import router as upload_router

app = FastAPI()
app.include_router(nutrition_router)
app.include_router(upload_router)