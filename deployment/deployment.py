import streamlit as st
import pandas as pd
import pickle
import warnings
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic

@st.cache_data
def load_location_data():
    df = pd.read_csv('dataset/cleaned_with_count.csv')

    location_mapping = {}
    location_display_list = []

    # Group by location detail and calculate averages
    for location_detail, group in df.groupby('Location Detail'):
        avg_lat = group['Latitude'].mean()
        avg_lng = group['Longitude'].mean()
        most_common_state = group['State'].mode()[0]  # Most frequent state
        # count = len(group)
        
        location_mapping[location_detail] = {
            'Latitude': avg_lat,
            'Longitude': avg_lng,
            'State': most_common_state,
            'Location Detail': location_detail
        }
        location_display_list.append(location_detail)
    
    unique_locations = sorted(location_display_list)
    return location_mapping, unique_locations

def get_nearby_similar_locations(df, selected_lat, selected_lng, predicted_price, price_margin=0.05, max_distance_km=2):
    similar_locations = []

    lower_bound = predicted_price * (1 - price_margin)
    upper_bound = predicted_price * (1 + price_margin)

    for _, row in df.iterrows():
        row_price = row['Price']
        if lower_bound <= row_price <= upper_bound:
            dist = geodesic((selected_lat, selected_lng), (row['Latitude'], row['Longitude'])).km
            if dist <= max_distance_km:
                similar_locations.append({
                    "Location Detail": row['Location Detail'],
                    "Price": row_price,
                    "Latitude": row['Latitude'],
                    "Longitude": row['Longitude']
                })

    return similar_locations

@st.cache_resource
def load_model_assets():
    warnings.filterwarnings("ignore", message=".*model_persistence.html.*")
    with open('model/models/models_result/best_model.pkl', 'rb') as f1, open('deployment/deployment_assets.pkl', 'rb') as f2:
        model = pickle.load(f1)
        assets = pickle.load(f2)
    return model, assets

def preprocess_user_input(user_data, assets):
    df_input = pd.DataFrame([user_data])
    
    # Encode categorical features
    for feature in assets['categorical_features']:
        if feature in df_input.columns:
            encoded_value = assets['encoders'][feature].transform(df_input[feature].astype(str))[0]
            df_input[feature + '_encoded'] = encoded_value
    
    # Select final features
    df_input = df_input[assets['final_features']]
    
    # Scale features
    df_scaled = assets['scaler'].transform(df_input)
    df_scaled = pd.DataFrame(df_scaled, columns=df_input.columns)
    
    return df_scaled

# --- Creating Streamlit APP ---
def render_title():
    st.title(" 🏠Room Rental Price Prediction")
 
def session_state_initialization_literal(literal, value):
    if literal not in st.session_state:
        st.session_state[literal] = value
 
def session_state_initialization_arr(arr):
    for key in arr:
        session_state_initialization_literal(key, False)
 
def render_location(location_list):
    session_state_initialization_literal("location", None)
    st.subheader("Search by location, property name, or city")
    st.selectbox(
        "Location",
        location_list,
        label_visibility="collapsed",
        key="location",
    )
 
def render_room_types(c):
    session_state_initialization_literal("room_type", None)
    c.subheader("Room type")
    room_options = ["Master Room", "Middle Room", "Single Room", "Soho", "Studio", "Suite"]
    c.radio(
        "Room type",
        room_options,
        label_visibility="collapsed",
        key="room_type",
    )
 
def render_status(c):
    session_state_initialization_literal("status", None)
    c.subheader("Tell us about yourself")
    status_options = ["Single Male", "Single Female", "Couple"]
    c.radio(
        "Status",
        status_options,
        label_visibility="collapsed",
        key="status",
    )
 
def render_nationality(c):
    session_state_initialization_literal("nationality", "Any")
    c.subheader("Nationality")
    nationality_options = ["Any", "Malaysian", "Non-Malaysian"]
    c.selectbox(
        "Nationality",
        nationality_options,
        label_visibility="collapsed",
        key="nationality",
    )
 
def render_race(c):
    c.subheader("Races")
    race_options = ["Malay", "Chinese", "Indian", "Other"]
    session_state_initialization_arr(race_options)
    for race in race_options:
        c.checkbox(race, key=race)
 
def render_occupation(c):
    c.subheader("Occupation")
    occupation_options = ["Student", "Employed", "Unemployed"]
    session_state_initialization_arr(occupation_options)
    for occupation in occupation_options:
        c.checkbox(occupation, key=occupation)
 
def render_preference(c):
    c.subheader("Preference")
    preference_options = ["Prefer muslim friendly", "Prefer pet allowance", "Prefer move-in immediately", "Prefer Zero Deposit"]
    session_state_initialization_arr(preference_options)
    for preference in preference_options:
        c.checkbox(preference, key=preference)
 
def render_lease_term(c):
    c.subheader("Lease Term")
    lease_term_options = ["< 6 months", "6 month", "12 months and above"]
    session_state_initialization_arr(lease_term_options)
    for lease_term in lease_term_options:
        c.checkbox(lease_term, key=lease_term)
 
def render_accomodation():
    st.subheader("Accommodations")
    cols = st.columns(3)
    accomodation_options = [
        "Near KTM / LRT / MRT",
        "Near KTM / LRT",
        "Near LRT / MRT",
        "Near KTM",
        "Near LRT",
        "Near MRT",
        "Near Train",
        "Near Bus stop",
        "24 hours security",
        "Swimming Pools",
        "Gymnasium Facility",
        "OKU Friendly",
        "Multi-purpose hall",
        "Playground",
        "Covered car park",
        "Surau",
        "Mini Market",
        "S&P",
        "Co-Living",
        "Air-Conditioning",
        "Washing Machine",
        "Wifi / Internet Access",
        "Cooking Allowed",
        "TV",
        "Share Bathroom",
        "Private Bathroom"
    ]
    session_state_initialization_arr(accomodation_options)
    for i, accomodation in enumerate(accomodation_options):
        cols[i % 3].checkbox(accomodation, key=accomodation)

def render_prediction(location_mapping, assets, model):
    # Initialize session state variables if they don't exist
    if 'is_predict_visible' not in st.session_state:
        st.session_state.is_predict_visible = False
    if 'prediction_message' not in st.session_state:
        st.session_state.prediction_message = ""

    if st.button("Predict"):
        # --- 1. Validation Phase ---
        validation_errors = []
        if not st.session_state.get("location"): # Use .get() for safer access
            validation_errors.append("⚠️ Please select a location to continue.")
        if not st.session_state.get("room_type"):
            validation_errors.append("⚠️ Please select a room type to continue.")
        if not st.session_state.get("status"):
            validation_errors.append("⚠️ Please select your status to continue.")
        # Add other necessary validations here

        if validation_errors:
            for error_msg in validation_errors:
                st.error(error_msg)
            st.session_state.is_predict_visible = False # Hide prediction section if errors
        else:
            # --- 2. All validations passed: Get user input and make prediction ---
            user_data = get_user_input(location_mapping, assets)
            preprocessed_input = preprocess_user_input(user_data, assets)

            # Call your actual prediction model/logic here
            predicted_price = model.predict(preprocessed_input) 

            # Store the formatted prediction message
            st.session_state.prediction_message = f"RM{predicted_price[0]:.2f}"
            st.session_state.is_predict_visible = True # Make prediction section visible
            st.session_state.user_data = user_data
            st.session_state.predicted_price = predicted_price[0]

    # --- 3. Display Phase (controlled by session state) ---
    if st.session_state.get('is_predict_visible', False):
        st.subheader("Predicted Rental Price")
        st.write("Based on the provided information, the predicted rental price is:")
        st.subheader(st.session_state.prediction_message) # Use markdown for formatting (e.g., bold)
        st.divider()

            # Load original dataset for reference
        df_all = pd.read_csv("dataset/cleaned_with_count.csv")

        # Retrieve user data from session
        user_data = st.session_state.get("user_data")
        predicted_price = st.session_state.get("predicted_price")

        if user_data is not None and predicted_price is not None:
            selected_lat = user_data["Latitude"]
            selected_lng = user_data["Longitude"]

        similar_places = get_nearby_similar_locations(df_all, selected_lat, selected_lng, predicted_price)

        if similar_places:
            st.subheader("Similar Room Rental Listings Nearby (2 km and ±5% of predicted price range)")
            m = folium.Map(location=[selected_lat, selected_lng], zoom_start=14)

            # Add marker for selected location (predicted location)
            folium.Marker(
                [selected_lat, selected_lng],
                popup="Selected Location",
                tooltip=f"Predicted Price: RM {predicted_price:.2f}",
                icon=folium.Icon(color="blue")
            ).add_to(m)
            
            # Add markers for similar listings
            for place in similar_places:
                folium.Marker(
                    [place["Latitude"], place["Longitude"]],
                    popup=f"{place['Location Detail']}<br>RM {place['Price']:.2f}",
                    tooltip=f"{place['Location Detail']} - RM {place['Price']:.2f}",
                    icon=folium.Icon(color="green")
                ).add_to(m)
        else:
            st.info("No similar listings found within 2 km and ±5% price range.")

    st.write("""
            This prediction is based on the current market trends and the characteristics of the room.
            For more accurate predictions, please consult a real estate professional or use a dedicated property rental platform.
            Thank you for using our Room Rental Price Prediction tool!
             """)


# def render_prediction():
#     def on_click():
#         st.session_state['is_predict_visible'] = True
 
#     if 'is_predict_visible' not in st.session_state:
#         st.session_state.is_predict_visible = False
 
#     st.button("Predict", on_click=on_click)
 
#     if st.session_state['is_predict_visible']:
#         st.subheader("Prediction Result")
#         st.write("Based on the provided information, the predicted rental price is RM 1,500 per month.")
#         st.write("This prediction is based on the current market trends and the characteristics of the room.")
#         st.write("For more accurate predictions, please consult a real estate professional or use a dedicated property rental platform.")
#         st.write("Thank you for using our Room Rental Price Prediction tool!")
#         st.divider()
 
 
def render_debug():
    if 'is_debug_visible' not in st.session_state:
        st.session_state.is_debug_visible = False
 
    st.button(
        "Toggle Debug Info",
        on_click=lambda: st.session_state.update({'is_debug_visible': not st.session_state['is_debug_visible']})
    )
 
    if st.session_state['is_debug_visible']:
        st.subheader("Debug Information")
        st.write(st.session_state)

def get_user_input(location_mapping, assets):
    user_data = dict()

    # Get coordinate and state for selected location
    selected_location = st.session_state["location"]
    location_info = location_mapping[selected_location]

    user_data["Location Detail"] = location_info["Location Detail"]
    user_data["State"] = location_info["State"]
    user_data["Longitude"] = location_info["Longitude"]
    user_data["Latitude"] = location_info["Latitude"]
    
    # Get selected room type
    user_data["Room Type"] = st.session_state["room_type"]

    # Get selected facility features
    for feature in assets['facility_features']:
        user_data[feature] = 1 if st.session_state[feature] else 0

    return user_data

def setup():
    # Load data and model
    location_mapping, unique_locations = load_location_data()
    model, assets = load_model_assets()

    render_title()
    render_location(unique_locations)
 
    row_1 = st.columns(3)
    render_room_types(row_1[0])
    render_status(row_1[1])
    render_nationality(row_1[2])
 
    row_2 = st.columns(4)
 
    render_race(row_2[0])
    render_occupation(row_2[1])
    render_preference(row_2[2])
    render_lease_term(row_2[3])
 
    render_accomodation()
    
    render_prediction(location_mapping, assets, model)
    # render_debug()
 
setup()
