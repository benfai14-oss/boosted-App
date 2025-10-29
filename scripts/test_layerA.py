
from ingestion.clients.weather_client import WeatherClient, RegionQuery
from ingestion.clients.market_client import MarketClient
from ingestion.clients.agri_client import AgriClient
from ingestion.transform.align import align_data
from ingestion.quality.contracts import SilverSchema

# paramètres
commodity = "corn"
regions = ["FR", "US"]
start_date = "2024-01-01"
end_date = "2024-03-31"

# Weather
weather = WeatherClient()
queries = [RegionQuery(r, start_date, end_date) for r in regions]
weather_df = weather.fetch_weather_anomalies(commodity, queries)
print("Weather data:")
print(weather_df.head())

# Market
market = MarketClient()
market_df = market.fetch_prices(commodity, start_date, end_date)
print("\nMarket data:")
print(market_df.head())

# Agri
agri = AgriClient()
agri_frames = {r: agri.fetch_production_stocks(r, start_date, end_date) for r in regions}
for r in regions:
    print(f"\nAgri data {r}:")
    print(agri_frames[r].head())

# Align / Silver layer
silver_df = align_data(weather_df, market_df, agri_frames, regions)
print("\nSilver layer:")
print(silver_df.head())

# Validation
SilverSchema.validate(silver_df)
print("\n✅ Silver layer validated with Pandera")
