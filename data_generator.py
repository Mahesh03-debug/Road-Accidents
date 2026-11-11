import random
import datetime
from pathlib import Path
import pandas as pd
from config import Config
from database import get_session, init_db
from models import Location, Accident

# Realistic regions and road corridors
SAMPLE_LOCATIONS = [
    {
        "region": "Downtown Commercial District",
        "area_type": "Urban",
        "roads": ["Main Street Arterial", "Central Market Blvd", "5th Cross Road", "Civic Center Way", None]
    },
    {
        "region": "North Ring Expressway",
        "area_type": "Highway",
        "roads": ["Express Tollway NH-44", "North Bypass Junction", "Ring Road Exit 12", "Elevated Freeway", None]
    },
    {
        "region": "Tech Corridor Sector 5",
        "area_type": "Suburban",
        "roads": ["Silicon Boulevard", "Software Park Link", "Cyber Junction Road", "Innovation Way", None]
    },
    {
        "region": "Industrial Estate South",
        "area_type": "Industrial",
        "roads": ["Freight Corridor 9", "Heavy Haul Road", "Warehouse Spur", "Factory Gate Avenue", None]
    },
    {
        "region": "Harbor Port Road",
        "area_type": "Highway",
        "roads": ["Port Access Parkway", "Container Terminal Way", "Dockside Expressway", None]
    },
    {
        "region": "Old City Historic Quarter",
        "area_type": "Urban",
        "roads": ["Heritage Market Lane", "Clock Tower Square", "Bazaar Main Road", "Temple Gate Way", None]
    },
    {
        "region": "West Suburbs Bypass",
        "area_type": "Suburban",
        "roads": ["Greenfield Parkway", "Valley Crossing Rd", "Suburban Link 14", "Orchard Highway", None]
    },
    {
        "region": "Airport Expressway Corridor",
        "area_type": "Highway",
        "roads": ["Aero Transit Freeway", "Terminal Flyover 3", "Airport North Approach", None]
    },
    {
        "region": "University Campus Perimeter",
        "area_type": "Urban",
        "roads": ["College Avenue", "Campus North Ring", "Research Boulevard", "Library Lane", None]
    },
    {
        "region": "East Riverfront District",
        "area_type": "Urban",
        "roads": ["Riverview Promenade", "Bridge Approach Road", "East Bank Drive", None]
    },
    {
        "region": "Metro Central Junction",
        "area_type": "Urban",
        "roads": ["Transit Interchange Plaza", "Station Road Underpass", "Grand Concourse", None]
    },
    {
        "region": "Rural Fringe Highway 10",
        "area_type": "Rural",
        "roads": ["State Highway 10 Mile 42", "Agrarian Crossing Road", "Village Connector 3", None]
    }
]

VEHICLE_WEIGHTS = {
    "Two-Wheeler": 0.35,
    "Passenger Car": 0.30,
    "Van / SUV": 0.12,
    "Commercial Truck": 0.10,
    "Bus": 0.05,
    "Auto Rickshaw": 0.05,
    "Bicycle / Pedestrian": 0.03
}

WEATHER_CHOICES = ["Clear", "Rain", "Fog / Mist", "Heavy Wind", "Overcast / Cloudy", None]
WEATHER_WEIGHTS = [0.55, 0.20, 0.10, 0.05, 0.05, 0.05]

LIGHT_CONDITIONS = ["Daylight", "Dusk / Dawn", "Dark - Streetlights On", "Dark - No Streetlights", None]
LIGHT_WEIGHTS = [0.50, 0.15, 0.22, 0.08, 0.05]

def generate_sample_dataset(count=1200, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Generates a realistic synthetic public road accident dataset.
    Reflects empirical traffic patterns:
    - Rush hour temporal spikes (8-10 AM, 5-8 PM)
    - Late night higher severity proportion
    - Two-wheelers / pedestrians suffering elevated severity
    - Weather-related collision clustering
    """
    if start_date is None:
        start_date = datetime.date(2025, 1, 1)
    if end_date is None:
        end_date = datetime.date(2025, 12, 31)

    total_days = (end_date - start_date).days
    records = []

    vehicles = list(VEHICLE_WEIGHTS.keys())
    v_weights = list(VEHICLE_WEIGHTS.values())

    for idx in range(1, count + 1):
        # 1. Date & Time generation with rush-hour modeling
        random_day_offset = random.randint(0, total_days)
        accident_date = start_date + datetime.timedelta(days=random_day_offset)

        # Weighted hours: rush hours (8-10, 17-20) have higher probabilities
        hour_weights = [
            1, 1, 1, 1, 2, 4, 7, 10,   # 00:00 - 07:00
            14, 15, 11, 9, 8, 8, 9, 11, # 08:00 - 15:00
            12, 16, 17, 14, 10, 7, 5, 3 # 16:00 - 23:00
        ]
        hour = random.choices(range(24), weights=hour_weights, k=1)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = datetime.datetime.combine(accident_date, datetime.time(hour, minute, second))

        # 2. Location
        loc_choice = random.choice(SAMPLE_LOCATIONS)
        region = loc_choice["region"]
        road = random.choice(loc_choice["roads"])
        area_type = loc_choice["area_type"]

        # 3. Vehicle Type
        vehicle_type = random.choices(vehicles, weights=v_weights, k=1)[0]

        # 4. Weather & Light
        weather = random.choices(WEATHER_CHOICES, weights=WEATHER_WEIGHTS, k=1)[0]
        light = random.choices(LIGHT_CONDITIONS, weights=LIGHT_WEIGHTS, k=1)[0]

        # 5. Severity distribution (Association modeling)
        # Base probabilities: Minor 55%, Moderate 27%, Severe 14%, Fatal 4%
        p_minor, p_mod, p_sev, p_fatal = 0.55, 0.27, 0.14, 0.04

        # Vulnerable road users: Two-wheelers & Pedestrians have higher severity
        if vehicle_type in ["Two-Wheeler", "Bicycle / Pedestrian"]:
            p_minor -= 0.15
            p_mod += 0.05
            p_sev += 0.07
            p_fatal += 0.03

        # Late night (23:00 - 04:00) speeds and low visibility elevate fatal/severe
        if hour in [23, 0, 1, 2, 3, 4]:
            p_minor -= 0.12
            p_mod -= 0.02
            p_sev += 0.08
            p_fatal += 0.06

        # Highway conditions elevate severe/fatal
        if area_type == "Highway":
            p_minor -= 0.10
            p_mod += 0.02
            p_sev += 0.05
            p_fatal += 0.03

        # Inclement weather (Rain, Fog) elevates severity
        if weather in ["Rain", "Fog / Mist"]:
            p_minor -= 0.05
            p_mod += 0.03
            p_sev += 0.02

        # Normalize weights
        raw_weights = [max(0.01, p) for p in [p_minor, p_mod, p_sev, p_fatal]]
        tot = sum(raw_weights)
        norm_weights = [w / tot for w in raw_weights]

        severity = random.choices(["Minor", "Moderate", "Severe", "Fatal"], weights=norm_weights, k=1)[0]

        # Casualties based on severity
        if severity == "Minor":
            casualties = random.choices([0, 1], weights=[0.85, 0.15], k=1)[0]
        elif severity == "Moderate":
            casualties = random.choices([1, 2, 3], weights=[0.70, 0.25, 0.05], k=1)[0]
        elif severity == "Severe":
            casualties = random.choices([1, 2, 3, 4], weights=[0.50, 0.35, 0.10, 0.05], k=1)[0]
        else: # Fatal
            casualties = random.choices([1, 2, 3, 5], weights=[0.65, 0.25, 0.07, 0.03], k=1)[0]

        # Context notes (Strictly non-PII anonymized)
        non_pii_note_templates = [
            "Side impact collision at intersection.",
            "Rear-end impact due to sudden braking in slow queue.",
            "Vehicle lost traction on wet pavement surface.",
            "Overtaking maneuver resulting in lateral scraping.",
            "Skid marks measured at scene, no secondary collision.",
            "Single-vehicle curb collision, lane barrier deflected impact.",
            "Pedestrian crossing event near designated crosswalk.",
            "Low-speed bumper collision during congestion crawl.",
            "Headlight glare reported as contributing environmental factor.",
            "Broadside collision during signal change transition.",
            ""
        ]
        notes = random.choice(non_pii_note_templates)

        records.append({
            "accident_id": idx,
            "accident_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "region": region,
            "road_name_optional": road,
            "area_type": area_type,
            "vehicle_type": vehicle_type,
            "severity": severity,
            "weather_optional": weather,
            "light_condition_optional": light,
            "casualty_count_optional": casualties,
            "notes_optional": notes
        })

    df = pd.DataFrame(records)
    return df

def seed_database(count=1200, force=False):
    """Initializes tables and seeds the database with synthetic data."""
    init_db(drop_all=force)
    session = get_session()

    try:
        from models import Location, Accident
        existing_accidents = session.query(Accident).count()
        if existing_accidents > 0 and not force:
            print(f"Database already populated with {existing_accidents} accidents. Use force=True to reseed.")
            return

        print(f"Generating {count} realistic accident records...")
        df = generate_sample_dataset(count=count)

        # 1. Create or retrieve unique locations
        unique_loc_combos = df[["region", "road_name_optional", "area_type"]].drop_duplicates()
        loc_map = {}

        for _, row in unique_loc_combos.iterrows():
            loc = Location(
                region=row["region"],
                road_name_optional=row["road_name_optional"] if pd.notna(row["road_name_optional"]) else None,
                area_type=row["area_type"]
            )
            session.add(loc)
            session.flush()
            key = (row["region"], str(row["road_name_optional"]) if pd.notna(row["road_name_optional"]) else "NONE")
            loc_map[key] = loc.location_id

        # 2. Insert accidents
        accident_objs = []
        for _, row in df.iterrows():
            key = (row["region"], str(row["road_name_optional"]) if pd.notna(row["road_name_optional"]) else "NONE")
            loc_id = loc_map[key]

            dt_parsed = datetime.datetime.strptime(row["accident_datetime"], "%Y-%m-%d %H:%M:%S")

            acc = Accident(
                accident_datetime=dt_parsed,
                location_id=loc_id,
                vehicle_type=row["vehicle_type"],
                severity=row["severity"],
                weather_optional=row["weather_optional"] if pd.notna(row["weather_optional"]) else None,
                light_condition_optional=row["light_condition_optional"] if pd.notna(row["light_condition_optional"]) else None,
                casualty_count_optional=int(row["casualty_count_optional"]),
                notes_optional=row["notes_optional"] if pd.notna(row["notes_optional"]) else None
            )
            accident_objs.append(acc)

        session.bulk_save_objects(accident_objs)
        session.commit()
        print(f"Successfully seeded {len(accident_objs)} accident records across {len(loc_map)} locations!")

        # Also save CSV to sample_data
        Config.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = Config.SAMPLE_DATA_DIR / "accident_records_sample.csv"
        df.to_csv(csv_path, index=False)
        print(f"Sample dataset exported to: {csv_path}")

    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        session.close()

if __name__ == "__main__":
    seed_database(count=1200, force=True)
