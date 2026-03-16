from fastapi        import FastAPI
from app.predictor  import predict
from app.schema     import CustomerData

app = FastAPI()

@app.post('/predict')
async def predict_value(customer: CustomerData):
    result = predict(customer)
    return result
    
    