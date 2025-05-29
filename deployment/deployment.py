import streamlit as st
import pandas as pd
import pickle
import warnings

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
    return location_mapping, unique_locations, df

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

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def find_nearby_similar_properties(
    selected_location_info, 
    predicted_price_selected, 
    all_locations_df, 
    model, 
    assets
):
    nearby_properties = []
    
    # This list MUST exactly match the 43 binary columns you provided that are used in the model
    # This ensures consistency for nearby property predictions
    model_binary_features = [
        "Air-Conditioning", "Washing Machine", "Wifi / Internet Access", 
        "Cooking Allowed", "TV", "Share Bathroom", "Private Bathroom",
        "Near KTM / LRT", "Near LRT / MRT", "Near KTM", "Near LRT", "Near MRT", 
        "Near Train", "Near Bus stop", "24 hours security", "Swimming Pools", 
        "Gymnasium Facility", "OKU Friendly", "Multi-purpose hall", "Playground", 
        "Covered car park", "Surau", "Mini Market", "Co-Living", "S&P",
        "Prefer muslim friendly", "Prefer move-in immediately", "< 6 month", 
        "6 month", "12 month and above", "Malaysian", "Student", "Employed", 
        "Unemployed", "Malay", "Chinese", "Indian", "Other", "Single Male", 
        "Single Female", "Couple", "Non-Malaysian", "Prefer Zero Deposit", 
        "Prefer pet allowed"
    ]

    # Iterate through all locations in the original dataframe
    for index, row in all_locations_df.iterrows():
        # Skip the selected location itself if it's the exact same entry
        # This check might need refinement if 'Location Detail' can have multiple entries
        # for the same lat/lng but different rooms etc.
        if (row['Latitude'] == selected_location_info['Latitude'] and 
            row['Longitude'] == selected_location_info['Longitude'] and
            row['Location Detail'] == selected_location_info['Location Detail']):
            continue

        distance = haversine(
            selected_location_info['Latitude'], 
            selected_location_info['Longitude'],
            row['Latitude'], 
            row['Longitude']
        )

        if distance <= 2:  # Within 2 km
            # Prepare data for prediction for the nearby property, only including model features
            nearby_user_data_raw = {
                "Location Detail": row["Location Detail"],
                "Longitude": row["Longitude"],
                "Latitude": row["Latitude"],
                "Room Type": row["Room Type"],
            }
            # Add all binary features directly from the row, ensuring they exist
            for feature in model_binary_features:
                nearby_user_data_raw[feature] = row.get(feature, 0) 

            # Preprocess using the same filtered logic as for user input
            preprocessed_nearby_input = preprocess_user_input(nearby_user_data_raw, assets)
            predicted_price_nearby = model.predict(preprocessed_nearby_input)[0]

            # Check if price is within +/- 5%
            price_lower_bound = predicted_price_selected * 0.95
            price_upper_bound = predicted_price_selected * 1.05

            if price_lower_bound <= predicted_price_nearby <= price_upper_bound:
                nearby_properties.append({
                    'latitude': row['Latitude'],
                    'longitude': row['Longitude'],
                    'Location Detail': row['Location Detail'],
                    'Predicted Price': predicted_price_nearby,
                    'Distance (km)': round(distance, 2)
                })
    return pd.DataFrame(nearby_properties)

# --- Creating Streamlit APP ---
def render_title():
    st.write("# Room Rental Price Prediction")
 
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

# --- The rest of the app setup ---
def render_prediction(location_mapping, assets, model, all_locations_df):
    # Initialize session state variables if they don't exist
    if 'is_predict_visible' not in st.session_state:
        st.session_state.is_predict_visible = False
    if 'prediction_message' not in st.session_state:
        st.session_state.prediction_message = ""
    if 'nearby_map_data' not in st.session_state:
        st.session_state.nearby_map_data = pd.DataFrame(columns=['latitude', 'longitude'])
    if 'predicted_price_selected' not in st.session_state:
        st.session_state.predicted_price_selected = 0.0 # Store the predicted price for selected location

    if st.button("Predict"):
        # --- 1. Validation Phase ---
        validation_errors = []
        if not st.session_state.get("location"):
            validation_errors.append("⚠️ Please select a location to continue.")
        if not st.session_state.get("room_type"):
            validation_errors.append("⚠️ Please select a room type to continue.")
        if not st.session_state.get("status"):
            validation_errors.append("⚠️ Please select your status to continue.")

        if validation_errors:
            for error_msg in validation_errors:
                st.error(error_msg)
            st.session_state.is_predict_visible = False
            st.session_state.nearby_map_data = pd.DataFrame(columns=['latitude', 'longitude']) # Clear map data
        else:
            # --- 2. All validations passed: Get user input and make prediction ---
            # get_user_input as per teammate's structure
            user_data_raw = get_user_input(location_mapping, assets) 
            
            # Preprocess: This will filter and format 'user_data_raw' for the model
            preprocessed_input = preprocess_user_input(user_data_raw, assets)
            predicted_price = model.predict(preprocessed_input)
            st.session_state.predicted_price_selected = predicted_price[0] # Store for later use

            st.session_state.prediction_message = f"""
            Based on the provided information, the predicted rental price is **RM{predicted_price[0]:.2f}**.

            This prediction is based on the current market trends and the characteristics of the room.
            For more accurate predictions, please consult a real estate professional or use a dedicated property rental platform.

            Thank you for using our Room Rental Price Prediction tool!
            """
            st.session_state.is_predict_visible = True

            # --- 3. Find and display nearby similar properties on a map ---
            selected_location_info = location_mapping[st.session_state["location"]]
            nearby_similar_buildings = find_nearby_similar_properties(
                selected_location_info, 
                st.session_state.predicted_price_selected, # Use the stored predicted price
                all_locations_df, 
                model, 
                assets
            )
            st.session_state.nearby_map_data = nearby_similar_buildings


    # --- 4. Display Phase (controlled by session state) ---
    if st.session_state.get('is_predict_visible', False):
        st.subheader("Prediction Result")
        st.markdown(st.session_state.prediction_message)
        st.divider()

        # Display map if nearby data exists
        if not st.session_state.nearby_map_data.empty:
            st.subheader("Nearby Similar Properties")
            # Highlight the selected location if desired
            selected_loc_df = pd.DataFrame([{
                'latitude': location_mapping[st.session_state["location"]]['Latitude'],
                'longitude': location_mapping[st.session_state["location"]]['Longitude'],
                'Location Detail': location_mapping[st.session_state["location"]]['Location Detail'],
                'Predicted Price': st.session_state.predicted_price_selected, # Use the stored predicted price
                'Distance (km)': 0 # Distance is 0 for the selected location
            }])
            
            # Combine selected location with nearby properties for map
            map_data = pd.concat([selected_loc_df, st.session_state.nearby_map_data])
            
            # st.map requires 'latitude' and 'longitude' columns
            st.map(map_data, latitude='latitude', longitude='longitude', zoom=13)
            
            st.write("Properties within 2km with +/- 5% similar predicted rental price:")
            st.dataframe(st.session_state.nearby_map_data[['Location Detail', 'Predicted Price', 'Distance (km)']].round(2), hide_index=True)
        else:
            st.info("No nearby similar properties found within 2km and +/- 5% rental price range.")

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
    location_mapping, unique_locations, all_data_df = load_location_data()
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
    
    render_prediction(location_mapping, assets, model, all_data_df)
    # render_debug()
 
setup()
