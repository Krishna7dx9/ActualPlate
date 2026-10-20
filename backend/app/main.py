from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.nutrition import router as nutrition_router
from backend.routes.upload import router as upload_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(nutrition_router)
app.include_router(upload_router)