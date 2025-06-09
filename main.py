from fastapi import FastAPI, HTTPException
from agent import generate_blog

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the BlogBot API!"}

@app.get("/generate_blog")
def get_blog(topic: str ):
    if not topic:
        raise HTTPException(status_code=400, detail="Topic query parameter is required.")
    try:
        blog_text = generate_blog(topic)
        return {"blog": blog_text}
    except Exception as e:
        return {"error": str(e)}
