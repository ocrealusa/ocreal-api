

```py
{
  "address": "1108 Veranda ct,Folsom,CA"
}
{
  "address": "2378 Captain Cook Dr, Anchorage, AK 99517"
}
# @app.post("/property/analytics", response_model=PropertyAnalyticsResponse)
# async def get_analytics_data(address_request: AddressAnalyticsRequest):
#     parsed_address = parse_address(address_request.address)
    
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         datafiniti_future = executor.submit(get_datafiniti_data, parsed_address)
#         rentcast_future = executor.submit(get_rentcast_data, parsed_address)

#         datafiniti_data, datafiniti_error = datafiniti_future.result()
#         rentcast_data, rentcast_error = rentcast_future.result()

#     errors = {}
#     if datafiniti_error:
#         errors['datafiniti'] = datafiniti_error
#     if rentcast_error:
#         errors['rentcast'] = rentcast_error

#     # Initialize default values
#     avg_price = None
#     price_per_sqft = None
#     lot_size = 1
#     historical_performance = []

#     # Process Datafiniti data if available
#     if isinstance(datafiniti_data, list) and len(datafiniti_data) > 0 and isinstance(datafiniti_data[0], dict):
#         prices = datafiniti_data[0].get('prices', [])
#         current_price = next((p['amountMax'] for p in prices if p.get('isSale') == 'false'), None)
#         price_per_sqft = next((p['pricePerSquareFoot'] for p in prices if 'pricePerSquareFoot' in p), None)
        
#         historical_prices = [p['amountMax'] for p in prices if 'amountMax' in p]
#         if historical_prices:
#             avg_price = sum(historical_prices) / len(historical_prices)
#         elif current_price:
#             avg_price = current_price

#         lot_size = datafiniti_data[0].get('lotSizeValue', 1)
#     else:
#         errors['datafiniti'] = errors.get('datafiniti', '') + " Data format unexpected or missing. Using estimates."

#     # Process Rentcast data if available
#     if isinstance(rentcast_data, dict) and 'history' in rentcast_data:
#         rentcast_history = rentcast_data['history']
#         historical_performance = [
#             {
#                 "month": datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%b"),
#                 "price": data['price']
#             }
#             for date, data in rentcast_history.items()
#         ]
#     else:
#         errors['rentcast'] = errors.get('rentcast', '') + " Data format unexpected or missing. Using estimates."

#     # TODO: REMOVE If we don't have real data, generate placeholder data
#     # if avg_price is None:
#     #     avg_price = 300000  # placeholder average price
#     #     errors['price'] = "Average price is an estimate due to missing data."

#     # if price_per_sqft is None:
#     #     price_per_sqft = avg_price / 1500  # assuming 1500 sqft as a placeholder
#     #     errors['price_per_sqft'] = "Price per square foot is an estimate due to missing data."

#     # if not historical_performance:
#     #     # Generate placeholder historical data
#     #     today = datetime.now()
#     #     historical_performance = [
#     #         {
#     #             "month": (today - timedelta(days=30*i)).strftime("%b"),
#     #             "price": int(avg_price * (1 + (random.random() - 0.5) * 0.1))  # +/- 5% variation
#     #         }
#     #         for i in range(12)
#     #     ]
#     #     errors['historical_performance'] = "Historical performance data is estimated due to missing data."

#     return PropertyAnalyticsResponse(
#         price={
#             "avg_price": round(avg_price, 2),
#             "per_sqrft": round(price_per_sqft, 2),
#             "per_acre": round(avg_price / lot_size, 2)
#         },
#         estimates={
#             "low_estimate": {
#                 "price": round(avg_price * 0.9, 2),
#                 "per_sqrft": round(price_per_sqft * 0.9, 2)
#             },
#             "high_estimate": {
#                 "price": round(avg_price * 1.1, 2),
#                 "per_sqrft": round(price_per_sqft * 1.1, 2)
#             }
#         },
#         historical_performance=historical_performance,
#         errors=errors
#     )
```
