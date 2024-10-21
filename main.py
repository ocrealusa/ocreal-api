from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import logging
import time
import json
from datetime import datetime
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a directory for API response logs
log_dir = "api_logs"
os.makedirs(log_dir, exist_ok=True)

# Load environment variables
load_dotenv(".env.local")

app = FastAPI(
    title="Home Value Estimator",
    description="An API for estimating home values and providing property analytics.",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATAFINITI_API_KEY = os.getenv("DATAFINITI_API_KEY")
RENTCAST_API_KEY = os.getenv("RENTCAST_API_KEY")
PRECISELY_API_KEY = os.getenv("PRECISELY_API_KEY")
PRECISELY_API_SECRET = os.getenv("PRECISELY_API_SECRET")
# Precisely API endpoint
PRECISELY_PROPERTY_ATTRIBUTES_URL = "https://api.precisely.com/property/v2/attributes/byaddress"

# API endpoints
DATAFINITI_URL = "https://api.datafiniti.co/v4/properties/search"
RENTCAST_URL = "https://api.rentcast.io/v1/avm/value"

CLOUDCMA_API_KEY = os.getenv("CLOUDCMA_API_KEY")
CLOUDCMA_URL = "https://cloudcma.com/properties/widget"

@app.get("/")
async def root():
    return {"message": "Welcome to the Home Value Estimator API 2024"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"An unexpected error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


class AddressEstimatorRequest(BaseModel):
    address: str
    user_estimate: float

class PropertyEstimatorResponse(BaseModel):
    user_estimate: float
    mspr: Optional[float]
    percentage_match: Optional[float]
    errors: dict

class AddressAnalyticsRequest(BaseModel):
    address: str

class PropertyAnalyticsResponse(BaseModel):
    price: dict
    estimates: dict
    historical_performance: List[dict]

class RentcastAPIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

class DatafinitiAPIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

def log_api_response(api_name: str, address: str, response_data: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{log_dir}/{api_name}_{address.replace(' ', '_')}_{timestamp}.json"
    # with open(filename, 'w') as f:
    #     json.dump(response_data, f, indent=2)
    logger.info(f"Logged {api_name} response for {address} ")

def get_datafiniti_data(address: str, max_retries: int = 3, retry_delay: int = 5):
    headers = {"Authorization": f"Bearer {DATAFINITI_API_KEY}"}
    data = {
        'query': f'address:{address}',
        'format': 'JSON',
        'num_records': 1,
        'download': False
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(DATAFINITI_URL, json=data, headers=headers)
            
            if response.status_code == 401:
                logger.error("Unauthorized access to Datafiniti API. Please check your API key.")
                return None, "Authentication failed"
            
            response.raise_for_status()
            data = response.json()
            
            # Log the API response
            log_api_response("Datafiniti", address, data)
            
            if 'records' in data and len(data['records']) > 0:
                return data['records'][0], None
            else:
                logger.warning(f"No records found. Retrying... (Attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        except requests.RequestException as e:
            logger.error(f"Error making API request: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {e}")
        except KeyError as e:
            logger.error(f"Unexpected response structure: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
            time.sleep(retry_delay)

    logger.error(f"Failed to retrieve valid data after {max_retries} attempts")
    return None, "Unable to retrieve data"

def get_rentcast_value_data(address: str):
    params = {
        "address": address,
        "propertyType": "Single Family",
        "bedrooms": 3,
        "bathrooms": 2,
        "squareFootage": 1500,
    }
    headers = {
        "accept": "application/json",
        "X-Api-Key": RENTCAST_API_KEY
    }

    try:
        response = requests.get(RENTCAST_URL, params=params, headers=headers)
        
        if response.status_code == 401:
            logger.error("Unauthorized access to Rentcast API. Please check your API key.")
            return None, "Authentication failed"
        elif response.status_code == 404:
            logger.error("Property not found in Rentcast database.")
            return None, "Property not found"
        
        response.raise_for_status()
        data = response.json()
        
        # Log the API response
        log_api_response("Rentcast", address, data)
        
        if not all(key in data for key in ['price', 'priceRangeLow', 'priceRangeHigh', 'pricePerSquareFoot', 'comparables']):
            raise ValueError("Incomplete data received from Rentcast API")
        return data, None
    except requests.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        return None, "Error fetching data"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None, "Unexpected error occurred"


@app.post("/property/estimator", response_model=PropertyEstimatorResponse)
async def get_property_data(address_request: AddressEstimatorRequest):
    with ThreadPoolExecutor(max_workers=1) as executor:
        datafiniti_future = executor.submit(get_datafiniti_data, address_request.address)
        datafiniti_data, datafiniti_error = datafiniti_future.result()

    errors = {}
    if datafiniti_error:
        errors['datafiniti'] = datafiniti_error

    datafiniti_estimate = None
    percentage_match = None

    if datafiniti_data:
        prices = datafiniti_data.get('prices', [])
        if prices:
            # Calculate the average of amountMax values
            valid_prices = [price['amountMax'] for price in prices if 'amountMax' in price]
            if valid_prices:
                datafiniti_estimate = sum(valid_prices) / len(valid_prices)

    if datafiniti_estimate and address_request.user_estimate:
        percentage_match = min(datafiniti_estimate, address_request.user_estimate) / max(datafiniti_estimate, address_request.user_estimate) * 100
        percentage_match = round(percentage_match, 2)

    logger.info(f"Estimate calculated for {address_request.address}: User estimate: {address_request.user_estimate}, API estimate: {datafiniti_estimate}, Match: {percentage_match}")

    return PropertyEstimatorResponse(
        user_estimate=address_request.user_estimate,
        mspr=round(float(datafiniti_estimate), 2) if datafiniti_estimate else 0,
        percentage_match=percentage_match if percentage_match else 0.0,
        errors=errors
    )

def get_rentcast_data(address: str):
    params = {
        "address": address,
        "propertyType": "Single Family",
        "bedrooms": 3,
        "bathrooms": 2,
        "squareFootage": 1500,
    }
    headers = {
        "accept": "application/json",
        "X-Api-Key": RENTCAST_API_KEY
    }

    try:
        response = requests.get(RENTCAST_URL, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        log_api_response("Rentcast", address, data)
        return data, None
    except requests.HTTPError as e:
        logger.error(f"Rentcast API error: {str(e)}")
        return None, f"Rentcast API error: {str(e)}"


def create_historical_entry(bedroom_count: int, price_value: float, date: datetime) -> dict:
    bedroom_mapping = {
        1: {"Studio": price_value, "1BD": price_value},
        2: {"2BD": price_value},
        3: {"3BD": price_value},
        4: {"4BD": price_value},
        5: {"5BD": price_value},
    }
    
    entry = {"month": date.strftime("%b")}
    entry.update(bedroom_mapping.get(bedroom_count, {}))
    return entry

import dateutil.parser

def parse_date(date_string: str) -> datetime:
    try:
        return dateutil.parser.isoparse(date_string)
    except ValueError:
        logger.error(f"Unable to parse date: {date_string}")
        return None

@app.post("/property/analytics", response_model=PropertyAnalyticsResponse)
async def get_analytics_data(address_request: AddressAnalyticsRequest):
    with ThreadPoolExecutor(max_workers=2) as executor:
        datafiniti_future = executor.submit(get_datafiniti_data, address_request.address)
        rentcast_future = executor.submit(get_rentcast_data, address_request.address)

        datafiniti_data, datafiniti_error = datafiniti_future.result()
        rentcast_data, rentcast_error = rentcast_future.result()

    if datafiniti_error and rentcast_error:
        raise HTTPException(status_code=400, detail=f"Datafiniti: {datafiniti_error}, Rentcast: {rentcast_error}")

    price_info = {
        "avg_price": None,
        "per_sqrft": None,
        "per_bedroom": None
    }
    estimates = {
        "low_estimate": {"price": None, "per_sqrft": None},
        "high_estimate": {"price": None, "per_sqrft": None}
    }
    historical_performance = []

    # Get bedroom count and square footage
    bedroom_count = datafiniti_data.get('numBedroom') if datafiniti_data else None
    square_footage = datafiniti_data.get('floorSizeValue') if datafiniti_data else None

    # Only process if it's a 1-5 bedroom house
    if bedroom_count and 1 <= bedroom_count <= 5:
        if rentcast_data:
            price_info["avg_price"] = round(rentcast_data.get("price", 0), 2)
            estimates["low_estimate"]["price"] = round(rentcast_data.get("priceRangeLow", 0), 2)
            estimates["high_estimate"]["price"] = round(rentcast_data.get("priceRangeHigh", 0), 2)

            if square_footage:
                price_info["per_sqrft"] = round(price_info["avg_price"] / square_footage, 2)
                estimates["low_estimate"]["per_sqrft"] = round(estimates["low_estimate"]["price"] / square_footage, 2)
                estimates["high_estimate"]["per_sqrft"] = round(estimates["high_estimate"]["price"] / square_footage, 2)

        if price_info["avg_price"] and bedroom_count:
            price_info["per_bedroom"] = round(price_info["avg_price"] / bedroom_count, 2)

        if datafiniti_data:
            prices = datafiniti_data.get('prices', [])
            if prices:
                for price in prices:
                    if 'dateSeen' in price and ('amountMax' in price or 'amountMin' in price):
                        price_value = price.get('amountMax') or price.get('amountMin', 0)
                        date_string = price['dateSeen'][0] if isinstance(price['dateSeen'], list) else price['dateSeen']
                        parsed_date = parse_date(date_string)
                        if parsed_date:
                            historical_entry = create_historical_entry(bedroom_count, price_value, parsed_date)
                            historical_performance.append(historical_entry)

    # Remove duplicate entries in historical_performance
    historical_performance = [dict(t) for t in {tuple(sorted(d.items())) for d in historical_performance}]

    return PropertyAnalyticsResponse(
        price=price_info,
        estimates=estimates,
        historical_performance=historical_performance
    )

class PropertyAttributesRequest(BaseModel):
    address: str
    attributes: Optional[str] = "all"

class PropertyAttributesResponse(BaseModel):
    property_attributes: Dict[str, Any]

def get_precisely_token():
    auth_url = "https://api.precisely.com/oauth/token"
    auth_data = {
        "grant_type": "client_credentials"
    }
    try:
        auth_response = requests.post(auth_url, auth=(PRECISELY_API_KEY, PRECISELY_API_SECRET), data=auth_data)
        logger.info(f"Token request status code: {auth_response.status_code}")
        logger.info(f"Token request response: {auth_response.text}")
        
        if auth_response.status_code == 200:
            token_data = auth_response.json()
            logger.info(f"Successfully obtained token: {token_data.get('access_token', '')[:10]}...")
            return token_data["access_token"]
        else:
            logger.error(f"Failed to obtain token. Status code: {auth_response.status_code}, Response: {auth_response.text}")
            raise HTTPException(status_code=401, detail=f"Failed to obtain Precisely API token. Status: {auth_response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Request exception in get_precisely_token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obtaining Precisely API token: {str(e)}")

@app.post("/property/attributes", response_model=PropertyAttributesResponse)
async def get_property_attributes(request: PropertyAttributesRequest):
    try:
        token = get_precisely_token()
        headers = {
            "Authorization": f"Bearer {token}"
        }
        params = {
            "address": request.address,
            "attributes": request.attributes
        }
        
        logger.info(f"Making request to Precisely API with params: {params}")
        response = requests.get(PRECISELY_PROPERTY_ATTRIBUTES_URL, headers=headers, params=params)
        logger.info(f"Precisely API response status code: {response.status_code}")
        logger.info(f"Precisely API response headers: {response.headers}")
        logger.info(f"Precisely API response content: {response.text[:500]}...")  # Log first 500 characters
        
        response.raise_for_status()
        data = response.json()
        
        # Log the API response
        log_api_response("Precisely", request.address, data)
        
        return PropertyAttributesResponse(property_attributes=data)
    except requests.HTTPError as e:
        logger.error(f"Precisely API HTTP error: {str(e)}")
        logger.error(f"Response content: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Precisely API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_property_attributes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


class CloudCMAReportRequest(BaseModel):
    mlsnum: str
    email_to: str

class CloudCMAReportResponse(BaseModel):
    success: bool
    message: str

@app.post("/property/cloudcma-report", response_model=CloudCMAReportResponse)
async def get_cloudcma_report(report_request: CloudCMAReportRequest):
    if not CLOUDCMA_API_KEY:
        raise HTTPException(status_code=500, detail="CloudCMA API key is not configured")

    data = {
        "api_key": CLOUDCMA_API_KEY,
        "mlsnum": report_request.mlsnum,
        "email_to": report_request.email_to
    }

    try:
        response = requests.post(CLOUDCMA_URL, data=data)
        response.raise_for_status()

        # Log the API response
        log_api_response("CloudCMA", f"MLS#{report_request.mlsnum}", response.json())

        return CloudCMAReportResponse(
            success=True,
            message="CloudCMA report request submitted successfully"
        )
    except requests.RequestException as e:
        logger.error(f"Error making CloudCMA API request: {e}")
        raise HTTPException(status_code=500, detail=f"Error requesting CloudCMA report: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)