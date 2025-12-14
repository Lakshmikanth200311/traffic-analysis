from fastapi import FastAPI
from app.routes.test_routes.yolo import router as yolo_router
from app.routes.test_routes.cycle_timing import router as cycle_timings

app = FastAPI(
    title="YOLO Traffic Detection API",
    description="API for vehicle detection using YOLOv8 on images and videos.",
    version="1.0",
)

app.include_router(yolo_router, prefix="/api", tags=["YOLO Detection"])
app.include_router(cycle_timings, prefix='/api', tags=['Cycle Calculations'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
