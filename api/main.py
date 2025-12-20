from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.upload_images import router as upload_router


def create_app() -> FastAPI:
	app = FastAPI(title="Quantum PhotoHub - HDR API", version="0.1.0")

	# CORS (adjust origins in production)
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_credentials=False,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	# Routers
	app.include_router(upload_router)

	return app


app = create_app()


if __name__ == "__main__":
	# Local dev server: uvicorn api.main:app --reload
	import uvicorn

	uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

