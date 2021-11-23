from app.main import app
import uvicorn

# This should be used only for debugging. Recommend using the uvicorn command in production.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)