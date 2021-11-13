from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Instantiate the app.
app = FastAPI()
 
@app.get("/")
async def hello():
    return {"endpoints": "Available endpoints: /predict"}

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

    class Config:
        schema_extra = {
            "example_data": {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            }
        }


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results


#uvicorn main:app --reload
# # By default, our app will be available locally at http://127.0.0.1:8000.
